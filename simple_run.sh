#!/bin/bash
#SBATCH --account=EB-MA3194M-019
#SBATCH --partition=spot-hbv2-120
#SBATCH --qos=spot-hbv2-120
#SBATCH --job-name=hbv2_demo
#SBATCH --output=logs/name_test_exp/test_exp_%A_%a.out
#SBATCH --error=logs/name_test_exp/test_exp_%A_%a.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=16
#SBATCH --time=00:05:00
#SBATCH --array=0-1

module purge
source /apps/build/easy_build/scripts/id_instance.sh
source /apps/build/easy_build/scripts/setup_modules.sh
module load Python/3.11.3-GCCcore-12.3.0
source /campaign/EB-MA3194M-019/big-venv-gpu/bin/activate

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK

python simple_run_experiment.py run --experiment alg_experiments.json --index $SLURM_ARRAY_TASK_ID
