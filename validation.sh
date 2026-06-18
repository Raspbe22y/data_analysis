#!/usr/bin/env bash
#SBATCH --job-name=test_inference
#SBATCH --partition=root
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --qos=short
#SBATCH --gres=gpu:1
#SBATCH --mem=40G
#SBATCH --time=12:00:00
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err

# ---- Environment setup ----
source /scratch_root/yg3023/miniconda3/etc/profile.d/conda.sh

# Activate your environment
conda activate datavenv

# ---- NCCL settings (for multi-GPU / multi-node) ----
export NCCL_DEBUG=WARN
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME="ib0,eno1,eth0"

# ---- PyTorch distributed setup ----
MASTER_ADDR=$(scontrol show hostnames "$SLURM_NODELIST" | head -n 1)
MASTER_PORT=29500

# ---- Run PyTorch distributed job ----
srun torchrun \
  --nnodes="$SLURM_NNODES" \
  --nproc_per_node="$SLURM_GPUS_ON_NODE" \
  --rdzv_backend=c10d \
  --rdzv_endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
  health_project/validation.py