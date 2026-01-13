#!/bin/bash
#SBATCH --account=EB-MA3194M-029
#SBATCH --partition=spot-ncv3-6
#SBATCH --qos=spot-ncv3-6
#SBATCH --job-name=pd3o-ag
#SBATCH --output=logs/pd3o-a/pd3o_%A_%a.out
#SBATCH --error=logs/pd3o-a/pd3o_%A_%a.err
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --time=01:30:00
#SBATCH --cpus-per-task=1
#SBATCH --array=0-0

module purge
source /apps/build/easy_build/scripts/id_instance.sh
source /apps/build/easy_build/scripts/setup_modules.sh
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.1.1
source /campaign/EB-MA3194M-029/lsmcode/test-venv-gpu/bin/activate

python run_experiment.py run --experiment pd3o_experiments.json --index $SLURM_ARRAY_TASK_ID
