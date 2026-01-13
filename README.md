# Kines-HPC: Deep Learning for Batted Ball Outcome Prediction on High-Performance Computing

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A tutorial project designed to introduce **Kinesiology and Sport Science professionals** to High-Performance Computing (HPC) and deep learning. This project demonstrates how to use GPU-accelerated computing on national supercomputing resources to train neural networks that predict baseball batted ball outcomes (out, single, double, triple, or home run) using MLB Statcast data.

## Table of Contents

- [Overview](#overview)
- [Why HPC for Kinesiology and Sport Science?](#why-hpc-for-kinesiology-and-sport-science)
- [Project Description](#project-description)
- [Performance Comparison](#performance-comparison)
- [Repository Structure](#repository-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Local Setup](#local-setup)
- [Accessing HPC Resources](#accessing-hpc-resources)
  - [What is ACCESS?](#what-is-access)
  - [Applying for an ACCESS Allocation](#applying-for-an-access-allocation)
  - [About SDSC Expanse](#about-sdsc-expanse)
- [Connecting to Expanse](#connecting-to-expanse)
  - [Setting Up SSH Keys](#setting-up-ssh-keys)
  - [Connecting via SSH](#connecting-via-ssh)
  - [Transferring Files](#transferring-files)
- [Running the Project on Expanse](#running-the-project-on-expanse)
  - [Launching an Interactive Jupyter Session](#launching-an-interactive-jupyter-session)
  - [Running the Notebook](#running-the-notebook)
- [Understanding the Model](#understanding-the-model)
- [Acknowledgments](#acknowledgments)
- [Resources](#resources)

---

## Overview

This project serves as a practical introduction to:

1. **Deep Learning Fundamentals** - Building and training a multi-layer perceptron (MLP) neural network using PyTorch
2. **High-Performance Computing** - Leveraging GPU resources on SDSC's Expanse supercomputer through the ACCESS program
3. **Sport Science Data Analysis** - Working with MLB Statcast data to predict batted ball outcomes

The goal is to demonstrate how researchers in Kinesiology and related fields can scale their data-intensive workflows using freely available national cyberinfrastructure resources.

---

## Why HPC for Kinesiology and Sport Science?

Modern sport science increasingly relies on large-scale data analysis:

- **Motion capture systems** generate millions of data points per session
- **Wearable sensors** produce continuous streams of biomechanical data
- **Video analysis** with computer vision requires significant computational power
- **Machine learning models** for performance prediction need extensive training

Training deep learning models on a laptop can take hours or even days. With HPC resources like GPU-enabled supercomputers, the same tasks can be completed in minutes, enabling:

- Faster iteration on model development
- Analysis of larger datasets
- More complex model architectures
- Hyperparameter optimization at scale

---

## Project Description

This project uses **MLB Statcast data from 2022-2025** to predict batted ball outcomes. When a ball is put into play, we classify the result into one of five categories:

| Class | Outcome | Description |
|-------|---------|-------------|
| 0 | Out | Field out, force out, double play, etc. |
| 1 | Single | Batter reaches first base |
| 2 | Double | Batter reaches second base |
| 3 | Triple | Batter reaches third base |
| 4 | Home Run | Batter rounds all bases |

### Features Used

The model uses 27 features including:

- **Batted ball characteristics**: Launch speed, launch angle, spray angle
- **Pitch characteristics**: Release speed, pitch type, location
- **Game situation**: Ball/strike count, outs, inning
- **Matchup information**: Batter stance, pitcher handedness

### Model Architecture

A 3-layer MLP (Multi-Layer Perceptron) with:
- Configurable hidden layer sizes (optimized via Optuna)
- Dropout regularization
- ReLU activation functions
- Class-weighted cross-entropy loss (to handle imbalanced outcomes)

---

## Performance Comparison

> **Note**: Fill in your actual timing results after running both locally and on Expanse.

| Environment | Hardware | Time per Epoch | Total Training Time (40 epochs) |
|-------------|----------|----------------|--------------------------------|
| Local (CPU) | _Your CPU model_ | _X min_ | _X hours_ |
| Expanse (GPU) | NVIDIA V100 (32GB) | _X sec_ | _X min_ |
| **Speedup** | | | **_X times faster_** |

### Resource Configuration Used on Expanse

- **Partition**: gpu-shared
- **GPUs**: 1x NVIDIA V100 (32GB)
- **CPUs**: 8 cores
- **Memory**: 64 GB
- **Time**: 6 hours

---

## Repository Structure

```
Kines-HPC/
├── README.md                    # This file
├── HPC_Outcomes.ipynb          # Main training notebook
├── data_collection.ipynb       # Notebook showing how Statcast data was collected
├── statcast_4years.csv         # Dataset (2022-2025 Statcast data)
├── requirements.txt            # Python dependencies
└── outputs/                    # Generated plots and model checkpoints
    ├── training_loss.png
    ├── validation_metrics.png
    └── confusion_matrix_test.png
```

---

## Getting Started

### Prerequisites

- Python 3.10 or higher
- Basic familiarity with:
  - Python programming
  - Command line / terminal
  - Jupyter notebooks

### Local Setup

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/Kines-HPC.git
   cd Kines-HPC
   ```

2. **Create a virtual environment** (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

4. **Run the notebook locally** (for baseline comparison):
   ```bash
   jupyter notebook HPC_Outcomes.ipynb
   ```

### requirements.txt

```
torch>=2.0.0
numpy>=1.21.0
pandas>=1.5.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
optuna>=3.0.0
pybaseball>=2.2.0
jupyter>=1.0.0
```

---

## Accessing HPC Resources

### What is ACCESS?

[ACCESS](https://access-ci.org/) (Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support) is an NSF-funded program that provides researchers with free access to national supercomputing resources. ACCESS replaced the former XSEDE program and offers computing time on systems across the country.

**Key benefits:**
- **Free** for U.S. researchers and educators
- No grant funding required
- Multiple allocation tiers for different needs
- Access to diverse computing resources (CPUs, GPUs, storage)

### Applying for an ACCESS Allocation

#### Step 1: Create an ACCESS Account

1. Go to [access-ci.org](https://access-ci.org/)
2. Click "Register" and create an account using your institutional email
3. Complete your profile and link your ORCID ID (optional but recommended)

#### Step 2: Request an Explore Allocation

For learning and small projects, the **Explore ACCESS** tier is ideal:

1. Log in to [allocations.access-ci.org](https://allocations.access-ci.org/)
2. Click "Prepare a New Request" → "Explore ACCESS"
3. Fill out the form:
   - **Project Title**: e.g., "Deep Learning for Sport Science Applications"
   - **Abstract**: Brief description of your intended work
   - **CV/Resume**: Upload your CV (max 3 pages)
4. Submit the request

**Explore ACCESS features:**
- Up to 400,000 ACCESS credits
- Approved within 1 business day
- No detailed proposal required
- Perfect for learning, benchmarking, and small research projects

#### Step 3: Exchange Credits for Resources

Once approved:

1. Go to your project in the ACCESS allocations portal
2. Click "Credits + Resources" → "Exchange"
3. Select "Expanse GPU" from the resource list
4. Enter the amount of credits to exchange
5. Provide a brief justification

**Exchange rates** (approximate):
- 1 GPU-hour on Expanse ≈ 1 ACCESS credit
- Check current rates at [allocations.access-ci.org](https://allocations.access-ci.org/)

### About SDSC Expanse

[Expanse](https://www.sdsc.edu/support/user_guides/expanse.html) is a supercomputer operated by the San Diego Supercomputer Center (SDSC) at UC San Diego.

**GPU Specifications:**
- 52 GPU nodes
- Each node: 4x NVIDIA V100 GPUs (32 GB each)
- Connected via NVLINK for fast GPU-to-GPU communication
- Dual 20-core Intel Xeon 6248 CPUs per node
- 256 GB DDR4 memory per node

**Why Expanse for this project:**
- Modern NVIDIA V100 GPUs ideal for deep learning
- Pre-configured PyTorch containers available
- Interactive Jupyter notebook support via Galyleo
- Strong user support and documentation

---

## Connecting to Expanse

### Setting Up SSH Keys

SSH keys provide secure, password-free authentication. Set them up once for convenient access.

#### On macOS/Linux:

1. **Generate an SSH key pair**:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@institution.edu"
   ```
   Press Enter to accept the default file location. Optionally set a passphrase.

2. **Copy your public key**:
   ```bash
   cat ~/.ssh/id_ed25519.pub
   ```
   Copy the entire output.

3. **Add the key to ACCESS**:
   - Log in to [access-ci.org](https://access-ci.org/)
   - Go to Profile → SSH Keys
   - Paste your public key and save

#### On Windows:

1. **Use Windows PowerShell or Git Bash**:
   ```bash
   ssh-keygen -t ed25519 -C "your_email@institution.edu"
   ```

2. **View your public key**:
   ```bash
   cat C:\Users\YourUsername\.ssh\id_ed25519.pub
   ```

3. **Add to ACCESS** as described above.

### Connecting via SSH

Once your SSH key is registered (allow 15-30 minutes for propagation):

```bash
ssh your_access_username@login.expanse.sdsc.edu
```

You should see the Expanse welcome message and command prompt.

### Transferring Files

#### Using SCP (Secure Copy):

```bash
# Upload a file to Expanse
scp local_file.csv your_username@login.expanse.sdsc.edu:/expanse/lustre/scratch/your_username/temp_project/

# Download a file from Expanse
scp your_username@login.expanse.sdsc.edu:/path/to/remote_file.csv ./local_directory/

# Upload an entire directory
scp -r local_folder/ your_username@login.expanse.sdsc.edu:/expanse/lustre/scratch/your_username/temp_project/
```

#### Using rsync (recommended for large transfers):

```bash
# Sync a directory (only transfers changed files)
rsync -avz --progress local_folder/ your_username@login.expanse.sdsc.edu:/expanse/lustre/scratch/your_username/temp_project/
```

#### Storage Locations on Expanse:

| Location | Path | Purpose | Quota |
|----------|------|---------|-------|
| Home | `/home/your_username` | Small files, scripts | 100 GB |
| Scratch | `/expanse/lustre/scratch/your_username/temp_project` | Large data, temporary | 10 TB |

**Important**: Scratch storage is purged periodically. Don't store important results there long-term.

---

## Running the Project on Expanse

### Launching an Interactive Jupyter Session

Expanse uses **Galyleo** to launch Jupyter notebooks on compute nodes. This gives you a notebook interface backed by GPU resources.

1. **SSH into Expanse**:
   ```bash
   ssh your_username@login.expanse.sdsc.edu
   ```

2. **Navigate to your project directory**:
   ```bash
   cd /expanse/lustre/scratch/your_username/temp_project
   ```

3. **Launch the Jupyter session**:
   ```bash
   /cm/shared/apps/sdsc/galyleo/galyleo launch \
     --account YOUR_ACCOUNT_ID \
     --partition gpu-shared \
     --nodes 1 \
     --cpus 8 \
     --memory 64 \
     --time-limit 06:00:00 \
     --gpus 1 \
     -e singularitypro \
     --bind /expanse,/scratch,/cvmfs \
     -s /cm/shared/apps/containers/singularity/pytorch/pytorch-latest.sif
   ```

   **Parameter explanations**:
   - `--account`: Your ACCESS allocation account (e.g., `css105`)
   - `--partition gpu-shared`: Use shared GPU partition (cost-effective)
   - `--nodes 1`: Request one compute node
   - `--cpus 8`: Request 8 CPU cores
   - `--memory 64`: Request 64 GB RAM
   - `--time-limit 06:00:00`: Maximum 6 hours runtime
   - `--gpus 1`: Request 1 GPU
   - `-s`: Path to the PyTorch Singularity container

4. **Access the notebook**:
   - Galyleo will output a URL (e.g., `https://...`)
   - Open this URL in your browser
   - Navigate to your notebook file

### Running the Notebook

1. Open `HPC_Outcomes.ipynb` in Jupyter
2. Verify GPU is available:
   ```python
   import torch
   print(f"CUDA available: {torch.cuda.is_available()}")
   print(f"Device: {torch.cuda.get_device_name(0)}")
   ```
3. Run all cells (Cell → Run All)
4. Monitor training progress and note timing for comparison

---

## Understanding the Model

### Data Pipeline

```
Raw Statcast Data (3M+ pitches)
         ↓
Filter to Balls in Play (~500K)
         ↓
Feature Engineering (27 features)
         ↓
Train/Val/Test Split (70/15/15)
         ↓
Preprocessing (Impute, Scale, One-Hot Encode)
         ↓
PyTorch DataLoaders
```

### Training Process

1. **Hyperparameter Optimization** (Optuna)
   - Searches for optimal hidden layer sizes, dropout rate, learning rate
   - 20 trials, 8 epochs each

2. **Final Training**
   - Uses best hyperparameters
   - 40 epochs with early stopping (patience=6)
   - Monitors validation macro-F1 score

3. **Evaluation**
   - Accuracy and macro-F1 on train/val/test sets
   - Confusion matrix visualization

### Key Techniques

- **Class Weighting**: Handles imbalanced data (many more outs than triples)
- **Early Stopping**: Prevents overfitting
- **Gradient Clipping**: Stabilizes training
- **AdamW Optimizer**: Modern optimizer with decoupled weight decay

---

## Acknowledgments

This project was developed as part of the [Cyberinfrastructure Professional (CIP) Fellows Program](https://www.sdsc.edu/education/CIP/CIP_fellow_program.html), a joint training program between:

- **San Diego Supercomputer Center (SDSC)** at UC San Diego
- **San Diego State University (SDSU)**
- **California State University, San Bernardino (CSUSB)**

The CIP Fellows Program is supported by NSF Award #2230127 and aims to train CI professionals with interdisciplinary skills to support scientific research teams.

**Computing Resources**: This work used the Expanse system at the San Diego Supercomputer Center through allocation [YOUR_ALLOCATION_ID] from the Advanced Cyberinfrastructure Coordination Ecosystem: Services & Support (ACCESS) program, which is supported by National Science Foundation grants #2138259, #2138286, #2138307, #2137603, and #2138296.

**Data Source**: MLB Statcast data accessed via the [pybaseball](https://github.com/jldbc/pybaseball) Python library.

---

## Resources

### ACCESS & HPC
- [ACCESS Documentation](https://access-ci.org/documentation/)
- [ACCESS Allocations Portal](https://allocations.access-ci.org/)
- [Expanse User Guide](https://www.sdsc.edu/support/user_guides/expanse.html)
- [SDSC Training Events](https://www.sdsc.edu/education_and_training/training.html)

### Deep Learning
- [PyTorch Tutorials](https://pytorch.org/tutorials/)
- [Optuna Documentation](https://optuna.readthedocs.io/)

### Statcast & Baseball Analytics
- [pybaseball Documentation](https://github.com/jldbc/pybaseball)
- [Baseball Savant](https://baseballsavant.mlb.com/)

### CIP Fellows Program
- [CIP Fellows Program Website](https://www.sdsc.edu/education/CIP/CIP_fellow_program.html)
- [CIP Fellows Portal](https://cip-fellows.sdsc.edu/)

---

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## Contact

For questions about this project or the CIP Fellows Program, please contact:

- **CIP Fellows Program**: [cip-fellows@sdsc.edu](mailto:cip-fellows@sdsc.edu)
- **SDSC User Support**: [help@sdsc.edu](mailto:help@sdsc.edu)

---

*Last updated: January 2026*
