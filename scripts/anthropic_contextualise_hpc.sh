#!/bin/bash
#SBATCH --job-name=conteb-cn7
#SBATCH --output=conteb-%J.out
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --nodelist=cn[9]
#SBATCH --gres=gpu:a30:1
#SBATCH --time=02:00:00
#SBATCH --mem=32000
#SBATCH --partition=scavenge

module purge # OTHERWISE IT WILL INHERIT THE MODULE AND BREAK EVERYTHING

set -e
set -x

module load CUDA/12.1.1
module load NCCL/2.18.3-GCCcore-12.3.0-CUDA-12.1.1
module load Miniconda3/25.5.1-1

export PYTHONNOUSERSITE=1

export CONDA_PKGS_DIRS=/home/macja/miniconda3-25-5-1-1/cache

# Load the conda shell function (required in batch jobs)
source /opt/itu/easybuild/software/Miniconda3/25.5.1-1/etc/profile.d/conda.sh

# Correct activation
conda activate /home/macja/miniconda3-25-5-1-1/envs/gpu311_pip_2


#run
cd ~/repositories/conteb-experiments
python3 scripts/evaluation/anthropic_repro.py \
    --save-dir "/home/macja/repositories/conteb-experiments/eval_conteb_repro/anthropic/run4_anthropic_mistakes" \
    --query-base-dir "illuin-conteb" \
    --documents-base-dir "/home/macja/repositories/conteb-experiments/contextualised_datasets" \
    --model-path "lightonai/modernbert-embed-large" \
    --use-prefix
