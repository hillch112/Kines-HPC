#!/usr/bin/env python
# coding: utf-8

# In[16]:


# In[17]:


import os
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline


import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

import optuna
from optuna.pruners import MedianPruner


# -------------------------
# Reproducibility
# -------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)


# In[18]:


# Optuna settings
N_TRIALS        = 30          # change as needed on HPC
N_EPOCHS_TUNE   = 12          # short runs for tuning
EPOCHS_FINAL    = 25          # longer run for final training

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
# Set file paths - UPDATE THESE FOR YOUR SETUP
DATA_PATH = "/expanse/lustre/scratch/chill2/temp_project/statcast_4years.csv"
OUTPUT_DIR = "/expanse/lustre/scratch/chill2/temp_project/outputs"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

# In[19]:


print("\nSECTION 1: LOADING DATA FROM CSV")
print("Looking for:", DATA_PATH)
print("Exists?", os.path.exists(DATA_PATH))

df = pd.read_csv(DATA_PATH)


print(f"Loaded data shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")


# In[20]:


print("\nSECTION 2: FILTERING TO BALLS IN PLAY (HITS + OUTS)")

# Define hit and out events
hit_types = ["single", "double", "triple", "home_run"]

out_event_types = [
    "field_out",
    "force_out",
    "double_play",
    "triple_play",
    "grounded_into_double_play",
    "other_out",
    "sac_fly",
    "sac_fly_double_play",
    "sac_bunt",
    "sac_bunt_double_play",
    "fielders_choice_out",
]

# Balls in play: type == 'X'
bip_mask = df["type"] == "X"
event_mask = df["events"].isin(hit_types + out_event_types)

data = df[bip_mask & event_mask].copy()

print(f"\n✓ Filtered to balls in play that are hits or outs")
print(f"  Rows: {len(data):,}")
print("\nEvent distribution:")
print(data["events"].value_counts().to_string())


# In[21]:


print("\nSECTION 3: BUILDING 5-OUTCOME TARGET")

# 5 outcome columns (one-hot style, mutually exclusive)
data["outs_in_play"] = data["events"].isin(out_event_types).astype(int)
data["single"]       = (data["events"] == "single").astype(int)
data["double"]       = (data["events"] == "double").astype(int)
data["triple"]       = (data["events"] == "triple").astype(int)
data["home_run"]     = (data["events"] == "home_run").astype(int)

outcome_cols = ["outs_in_play", "single", "double", "triple", "home_run"]

# Sanity check
row_sum = data[outcome_cols].sum(axis=1)
assert (row_sum == 1).all(), "Some rows do not map to exactly one of the 5 outcomes!"

print("\n✓ Created 5 outcome columns:")
print(data[outcome_cols].head())
print("\nOutcome counts:")
print(data[outcome_cols].sum())
print("\nOutcome proportions (%):")
print((data[outcome_cols].mean() * 100).round(2))

# Build a single 5-class label: 0–4
# 0 = outs_in_play, 1 = single, 2 = double, 3 = triple, 4 = home_run
conditions = [
    data["outs_in_play"] == 1,
    data["single"] == 1,
    data["double"] == 1,
    data["triple"] == 1,
    data["home_run"] == 1,
]
choices = [0, 1, 2, 3, 4]
data["outcome_class"] = np.select(conditions, choices, default=-1).astype(int)
assert (data["outcome_class"] >= 0).all(), "Found rows with invalid outcome_class!"

print("\nOutcome_class distribution (0=out,1=1B,2=2B,3=3B,4=HR):")
print(data["outcome_class"].value_counts().sort_index())


# In[22]:


print("\nSECTION 4: CREATING NEW FEATURES")

print("\nCreating derived features...")

# spray_angle: direction of batted ball using home plate reference (125.42, 125.42)
if "hc_x" in data.columns and "hc_y" in data.columns:
    data["spray_angle"] = (
        np.arctan2(data["hc_y"] - 125.42, data["hc_x"] - 125.42) * 180 / np.pi
    )
    print("  ✓ spray_angle: Direction of batted ball")

# horizontal_distance: how far left/right the ball went
if "hc_x" in data.columns:
    data["horizontal_distance"] = np.abs(data["hc_x"] - 125.42)
    print("  ✓ horizontal_distance: |hc_x - center|")

# pitch_distance_from_center: distance from middle of the strike zone
if "plate_x" in data.columns and "plate_z" in data.columns:
    data["pitch_distance_from_center"] = np.sqrt(
        data["plate_x"] ** 2 + (data["plate_z"] - 2.5) ** 2
    )
    print("  ✓ pitch_distance_from_center: distance from zone center")

# count: ball-strike count (categorical)
if "balls" in data.columns and "strikes" in data.columns:
    data["count"] = data["balls"].astype(str) + "-" + data["strikes"].astype(str)
    print("  ✓ count: 'balls-strikes' representation")

# runners_on: total number of baserunners
runner_columns = ["on_1b", "on_2b", "on_3b"]
if all(col in data.columns for col in runner_columns):
    data["runners_on"] = (~data[runner_columns].isna()).sum(axis=1)
    print("  ✓ runners_on: total base runners")

# launch_speed_x_angle: interaction between EV and LA
if "launch_speed" in data.columns and "launch_angle" in data.columns:
    data["launch_speed_x_angle"] = data["launch_speed"] * data["launch_angle"]
    print("  ✓ launch_speed_x_angle: launch_speed * launch_angle")

print(f"\n✓ Feature engineering complete")
print(f"  Total columns now: {len(data.columns)}")


# In[23]:


print("\nSECTION 5: SELECTING RELEVANT FEATURES FOR MODELING")

# Define the features you want to use (no 'events', no outcome columns here)
selected_features = [
    # Batted ball characteristics
    "launch_speed",      # Exit velocity in mph
    "launch_angle",      # Angle off the bat in degrees
    "hit_distance_sc",   # Projected hit distance
    "bb_type",           # Batted ball type (fly_ball, ground_ball, etc.)
    'attack_angle',
    'attack_direction',

    # Hit location
    "hc_x",              # Hit coordinate X
    "hc_y",              # Hit coordinate Y

    # Pitch characteristics
    "release_speed",     # Pitch velocity
    "pitch_type",        # Type of pitch (FF, SL, CH, etc.)
    "plate_x",           # Horizontal pitch location
    "plate_z",           # Vertical pitch location
    'arm_angle',
    'release_spin_rate',
    'spin_axis',

    # Game situation
    "balls",             # Ball count
    "strikes",           # Strike count
    "outs_when_up",      # Number of outs
    "inning",            # Inning number

    # Matchup information
    "stand",             # Batter stance (L/R)
    "p_throws",          # Pitcher handedness (L/R)


    # Engineered features
    "spray_angle",
    "horizontal_distance",
    "pitch_distance_from_center",
    "count",
    "runners_on",
    "launch_speed_x_angle",
]

# Keep only those features that actually exist in `data`
available_features = [col for col in selected_features if col in data.columns]
X = data[available_features].copy()
y = data["outcome_class"].values.astype(np.int64)

print(f"\n✓ Selected {len(available_features)} features")
print(f"  Features included: {', '.join(available_features)}")

# Identify categorical and numerical features
categorical_features = X.select_dtypes(include=["object"]).columns.tolist()
numerical_features   = X.select_dtypes(include=[np.number]).columns.tolist()

print(f"\nCategorical features ({len(categorical_features)}): {categorical_features}")
print(f"Numeric features ({len(numerical_features)}): {numerical_features}")
print(f"\nTarget (outcome_class) shape: {y.shape}")


# In[24]:


print("\nSECTION 6: TRAIN/VAL/TEST SPLIT")



# First: train+val vs test
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y,
    test_size=0.15,      # 15% test
    random_state=42,
    stratify=y
)

# Then: split temp into train and val
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp,
    test_size=0.1765,    # 0.1765 * 0.85 ≈ 0.15 → 70/15/15 overall
    random_state=42,
    stratify=y_temp
)

print("Split sizes:")
print("  X_train:", X_train.shape)
print("  X_val:  ", X_val.shape)
print("  X_test: ", X_test.shape)


# In[25]:


print("\nSECTION 7: BUILDING PREPROCESSING PIPELINE (IMPUTE + SCALE/OHE)")

categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()
numeric_features     = X_train.select_dtypes(include=[np.number]).columns.tolist()

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),  # or "constant", fill_value="MISSING"
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse=False)),
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ],
    remainder="drop"
)

X_train_proc = preprocessor.fit_transform(X_train)
X_val_proc   = preprocessor.transform(X_val)
X_test_proc  = preprocessor.transform(X_test)

print("\nPreprocessed shapes:")
print("  X_train_proc:", X_train_proc.shape)
print("  X_val_proc:  ", X_val_proc.shape)
print("  X_test_proc: ", X_test_proc.shape)

input_dim = X_train_proc.shape[1]
print(f"\nModel input dimension: {input_dim}")


# In[26]:


print("\nSECTION 8: HYPERPARAMETER SEARCH WITH OPTUNA")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# ----- Tensors & Dataloaders -----

X_train_t = torch.tensor(X_train_proc, dtype=torch.float32)
X_val_t   = torch.tensor(X_val_proc,   dtype=torch.float32)
X_test_t  = torch.tensor(X_test_proc,  dtype=torch.float32)

y_train_t = torch.tensor(y_train, dtype=torch.long)
y_val_t   = torch.tensor(y_val,   dtype=torch.long)
y_test_t  = torch.tensor(y_test,  dtype=torch.long)

BATCH_SIZE = 1024

train_loader = DataLoader(
    TensorDataset(X_train_t, y_train_t),
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False
)
val_loader = DataLoader(
    TensorDataset(X_val_t, y_val_t),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)
test_loader = DataLoader(
    TensorDataset(X_test_t, y_test_t),
    batch_size=BATCH_SIZE,
    shuffle=False,
    drop_last=False
)

# ----- Model definition -----

class MLP5(nn.Module):
    def __init__(self, in_dim: int, hidden_dims=None, dropout: float = 0.2, num_classes: int = 5):
        super().__init__()
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]

        layers = []
        prev = in_dim
        for h in hidden_dims:
            layers.append(nn.Linear(prev, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev = h

        layers.append(nn.Linear(prev, num_classes))  # logits
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# ----- Evaluation helper -----

from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

def evaluate_multiclass(loader, mdl):
    mdl.eval()
    all_y = []
    all_pred = []
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)
            logits = mdl(xb)
            preds = torch.argmax(logits, dim=1)
            all_y.append(yb.cpu().numpy())
            all_pred.append(preds.cpu().numpy())
    y_true = np.concatenate(all_y)
    y_pred = np.concatenate(all_pred)

    acc = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)
    return {"acc": acc, "macro_f1": macro_f1, "y_true": y_true, "y_pred": y_pred}


print("\nRunning Optuna hyperparameter search...")

def objective(trial):
    # ---- Hyperparameters to search ----
    hidden1 = trial.suggest_int("hidden1", 128, 512)
    hidden2 = trial.suggest_int("hidden2", 64, 512)
    hidden3 = trial.suggest_int("hidden3", 32, 256)
    dropout = trial.suggest_float("dropout", 0.0, 0.5)
    lr = trial.suggest_float("lr", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-3, log=True)

    # ---- Build model for this trial ----
    model = MLP5(
        in_dim=X_train_proc.shape[1],
        hidden_dims=[hidden1, hidden2, hidden3],
        dropout=dropout,
        num_classes=5
    ).to(DEVICE)

    # Class weights for imbalance (computed from train labels)
    class_counts = np.bincount(y_train)
    num_classes_ = len(class_counts)
    class_weights = (class_counts.sum() / (num_classes_ * class_counts)).astype(np.float32)
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    criterion = nn.CrossEntropyLoss(weight=class_weights_t)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

    EPOCHS_SEARCH = 8  # small number of epochs per trial to keep search feasible

    for epoch in range(EPOCHS_SEARCH):
        model.train()
        for xb, yb in train_loader:
            xb = xb.to(DEVICE)
            yb = yb.to(DEVICE)

            optimizer.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

    # Evaluate on validation set
    metrics = evaluate_multiclass(val_loader, model)
    val_macro_f1 = metrics["macro_f1"]

    # We want to maximize macro-F1
    return val_macro_f1

study = optuna.create_study(direction="maximize")
study.optimize(objective, n_trials=20)  # bump this on HPC (e.g., 50–200)

print("\nBest trial found by Optuna:")
print("  Value (val macro-F1):", study.best_trial.value)
print("  Params:", study.best_trial.params)

best_params = study.best_params

print("\nTraining final model with best hyperparameters...")

model = MLP5(
    in_dim=X_train_proc.shape[1],
    hidden_dims=[
        best_params["hidden1"],
        best_params["hidden2"],
        best_params["hidden3"],
    ],
    dropout=best_params["dropout"],
    num_classes=5,
).to(DEVICE)

# Recompute class weights for final training
class_counts = np.bincount(y_train)
num_classes_ = len(class_counts)
class_weights = (class_counts.sum() / (num_classes_ * class_counts)).astype(np.float32)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"],
)

EPOCHS = 40
PATIENCE = 6
best_val_f1 = -np.inf
best_state = None
pat = PATIENCE

history_epochs = []
history_train_loss = []
history_val_acc = []
history_val_macro_f1 = []


for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    train_loss = running_loss / max(n_batches, 1)
    val_metrics = evaluate_multiclass(val_loader, model)

    print(
        f"Epoch {epoch:03d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_acc={val_metrics['acc']:.4f} | "
        f"val_macro_f1={val_metrics['macro_f1']:.4f}"
    )

    history_epochs.append(epoch)
    history_train_loss.append(train_loss)
    history_val_acc.append(val_metrics["acc"])
    history_val_macro_f1.append(val_metrics["macro_f1"])

    # Early stopping on validation macro-F1
    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        best_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
        }
        pat = PATIENCE
    else:
        pat -= 1
        if pat <= 0:
            print("Early stopping triggered.")
            break

# Load best model state
if best_state is not None:
    model.load_state_dict(best_state["model_state"])
    print(f"\nLoaded best model from epoch {best_state['epoch']} with val_macro_f1={best_val_f1:.4f}")
else:
    print("\nWarning: best_state is None; using last epoch model.")


# In[27]:


print("\nSECTION 9: FINAL TRAINING WITH BEST HYPERPARAMETERS")

print("\nTraining final model with best hyperparameters...")

model = MLP5(
    in_dim=X_train_proc.shape[1],
    hidden_dims=[
        best_params["hidden1"],
        best_params["hidden2"],
        best_params["hidden3"],
    ],
    dropout=best_params["dropout"],
    num_classes=5,  # or n_classes
).to(DEVICE)

# Recompute class weights for final training
class_counts = np.bincount(y_train)
num_classes_ = len(class_counts)
class_weights = (class_counts.sum() / (num_classes_ * class_counts)).astype(np.float32)
class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

criterion = nn.CrossEntropyLoss(weight=class_weights_t)
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=best_params["lr"],
    weight_decay=best_params["weight_decay"],
)

EPOCHS = 40
PATIENCE = 6
best_val_f1 = -np.inf
best_state = None
pat = PATIENCE

history_epochs = []
history_train_loss = []
history_val_acc = []
history_val_macro_f1 = []

for epoch in range(1, EPOCHS + 1):
    model.train()
    running_loss = 0.0
    n_batches = 0

    for xb, yb in train_loader:
        xb = xb.to(DEVICE)
        yb = yb.to(DEVICE)

        optimizer.zero_grad(set_to_none=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        running_loss += loss.item()
        n_batches += 1

    train_loss = running_loss / max(n_batches, 1)

    # Validation metrics
    val_metrics = evaluate_multiclass(val_loader, model)

    print(
        f"Epoch {epoch:03d} | "
        f"train_loss={train_loss:.4f} | "
        f"val_acc={val_metrics['acc']:.4f} | "
        f"val_macro_f1={val_metrics['macro_f1']:.4f}"
    )

    history_epochs.append(epoch)
    history_train_loss.append(train_loss)
    history_val_acc.append(val_metrics["acc"])
    history_val_macro_f1.append(val_metrics["macro_f1"])

    # Early stopping on validation macro-F1
    if val_metrics["macro_f1"] > best_val_f1:
        best_val_f1 = val_metrics["macro_f1"]
        best_state = {
            "epoch": epoch,
            "model_state": model.state_dict(),
        }
        pat = PATIENCE
    else:
        pat -= 1
        if pat <= 0:
            print("Early stopping triggered.")
            break

# Load best model state
if best_state is not None:
    model.load_state_dict(best_state["model_state"])
    print(f"\nLoaded best model from epoch {best_state['epoch']} with val_macro_f1={best_val_f1:.4f}")
else:
    print("\nWarning: best_state is None; using last epoch model.")


# In[28]:


print("\nSECTION 10: FINAL EVALUATION")

def evaluate_split(name, loader, model):
    metrics = evaluate_multiclass(loader, model)
    acc = metrics["acc"]
    f1 = metrics["macro_f1"]
    preds = metrics["y_pred"]
    labels = metrics["y_true"]
    print(f"{name} -> Acc: {acc:.3f}, Macro F1: {f1:.3f}")
    return acc, f1, preds, labels

train_metrics = evaluate_split("Train", train_loader, model)
val_metrics   = evaluate_split("Val",   val_loader,   model)
test_metrics  = evaluate_split("Test",  test_loader,  model)


# In[29]:


print("\nSECTION 11: VISUALIZATIONS")

# 8.1 Loss curve
plt.figure()
plt.plot(history_epochs, history_train_loss, label="Train Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig("training_loss.png", dpi=150)
plt.close()

# 8.2 Validation metrics curves
plt.figure()
plt.plot(history_epochs, history_val_acc, label="Val Acc")
plt.plot(history_epochs, history_val_macro_f1, label="Val Macro F1")
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Validation Accuracy & Macro F1")
plt.legend()
plt.tight_layout()
plt.savefig("validation_metrics.png", dpi=150)
plt.close()

# 8.3 Confusion matrix on test
_, _, test_preds, test_labels = test_metrics
cm = confusion_matrix(test_labels, test_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
plt.figure()
disp.plot(values_format="d")
plt.title("Confusion Matrix - Test Set")
plt.tight_layout()
plt.savefig("confusion_matrix_test.png", dpi=150)
plt.close()

print("Saved plots:")
print("  training_loss.png")
print("  validation_metrics.png")
print("  confusion_matrix_test.png")

print("\nAll done.")


# In[ ]:

print("Training complete!")
print(f"Results saved to: {OUTPUT_DIR}")


