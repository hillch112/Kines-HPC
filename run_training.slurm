#!/bin/bash
#SBATCH --job-name="kines-hpc"
#SBATCH --output="kines-hpc.%j.%N.out"
#SBATCH --error="kines-hpc.%j.%N.err"
#SBATCH --partition=gpu-shared
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --account=YOUR_ACCOUNT_ID
#SBATCH --time=06:00:00
#SBATCH --export=ALL

# ============================================================
# SLURM BATCH SCRIPT FOR KINES-HPC
# ============================================================
# This script submits a batch job to train the neural network
# on Expanse's GPU nodes using a Singularity container.
#
# BEFORE RUNNING:
# 1. Replace YOUR_ACCOUNT_ID with your ACCESS allocation account
# 2. Update the project directory path if needed
# 3. Make sure train_model.py is in your project directory
#
# USAGE:
#   sbatch run_training.slurm
#
# MONITOR:
#   squeue -u $USER
#
# CANCEL:
#   scancel <job_id>
# ============================================================

echo "========================================"
echo "KINES-HPC Training Job"
echo "========================================"
echo "Job started: $(date)"
echo "Job ID: $SLURM_JOB_ID"
echo "Node: $SLURMD_NODENAME"
echo "Working directory: $(pwd)"
echo "========================================"

# Change to your project directory
cd /expanse/lustre/scratch/$USER/temp_project

# Print environment info
echo ""
echo "Environment Information:"
echo "------------------------"
echo "User: $USER"
echo "Hostname: $(hostname)"
echo "Current directory: $(pwd)"
echo ""

# Load the Singularity module
module purge
module load singularitypro

# Define the container path
CONTAINER="/cm/shared/apps/containers/singularity/pytorch/pytorch-latest.sif"

echo "Container: $CONTAINER"
echo ""

# Check if the training script exists
if [ ! -f "train_model.py" ]; then
    echo "ERROR: train_model.py not found in $(pwd)"
    echo "Please make sure the training script is in your project directory."
    exit 1
fi

echo "Starting training..."
echo "========================================"
echo ""

# Run the training script inside the container
# --nv enables NVIDIA GPU support
# --bind mounts the necessary filesystems
singularity exec --nv \
    --bind /expanse,/scratch,/cvmfs \
    $CONTAINER \
    python train_model.py

# Capture the exit status
EXIT_STATUS=$?

echo ""
echo "========================================"
echo "Job finished: $(date)"
echo "Exit status: $EXIT_STATUS"
echo "========================================"

exit $EXIT_STATUS
