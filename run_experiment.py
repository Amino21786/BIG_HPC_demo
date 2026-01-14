"""Generate the experiments list"""

import argparse
from itertools import product
import json
from pathlib import Path
from typing import Dict, List
from process import process

def save_slurm_script(path: Path, ntasks: int, experiment_path: Path):
    """
    Save the slurm script to path for a job array with ntasks tasks.

    Parameters
    ----------
    path : Path
        Path to save the slurm script.
    ntasks : int
        Number of tasks in the job array.
    experiment_path : Path
        Path to the experiments list JSON file.
    """

    ## to be edited
    with open(path, "w") as txt:
        txt.write("#!/bin/bash\n")
        txt.write("#SBATCH --account=EB-MA3194M-019\n") 
        txt.write("#SBATCH --partition=spot-ncv3-6\n") # type of instance (spot, paygo on CPU or GPU)
        txt.write("#SBATCH --qos=spot-ncv3-6\n") # cpu or gpu instance 
        txt.write("#SBATCH --job-name=name_test_exp\n")
        txt.write("#SBATCH --output=logs/name_test_exp/test_exp_%A_%a.out\n")
        txt.write("#SBATCH --error=logs/name_test_exp/test_exp_%A_%a.err\n")
        txt.write("#SBATCH --nodes=1\n")
        txt.write("#SBATCH --gres=gpu:1\n") # gpu or cpu
        txt.write("#SBATCH --time=00:05:00\n")
        txt.write("#SBATCH --cpus-per-task=1\n")
        txt.write(f"#SBATCH --array=0-{ntasks-1}\n\n")

        # setup to load modules and virtual environment (Python, CUDA)
        txt.write("module purge\n")
        txt.write("source /apps/build/easy_build/scripts/id_instance.sh\n")
        txt.write("source /apps/build/easy_build/scripts/setup_modules.sh\n")
        txt.write("module load Python/3.11.3-GCCcore-12.3.0\n")
        txt.write("module load CUDA/12.1.1\n")
        txt.write("source /campaign/EB-MA3194M-019/big-venv-gpu/bin/activate\n\n") #starts virtual environment in the HPC

        txt.write(
            f"python run_experiment.py run --experiment {experiment_path} --index $SLURM_ARRAY_TASK_ID\n"
        )

def create_experiments_list(params):
    """
    Create the experiments list and save it to experiment_path.

    Parameters
    ----------
    params : dict
        Parameters for the experiments.

    Returns
    -------
    int
        Number of experiments created.
    """

    experiments = []
    for case in params:
        algo = case["algorithm"]
        case.pop("algorithm")  # Remove algorithm from the grid search
        keys = case.keys()
        values = case.values()
        for combination in product(*values):
            experiment = {"algorithm": algo, **dict(zip(keys, combination))}
            # experiment["algorithm"] = algo
            # experiment = dict(zip(keys, combination))
            experiments.append(experiment)

    print(f"Created {len(experiments)} experiments.")

    return experiments


def save_experiments_list(path: Path, experiments: List[Dict]):
    """Save experiments list to a JSON file.

    Parameters
    ----------
    path : Path
        Path to the JSON file.
    experiments : List[dict]
        Experiments list to save.
    """
    with open(path, "w") as f:
        json.dump(experiments, f, indent=4)


def configure(args):
    """Main function to create experiments list and slurm script."""
    print(f"Loading parameters from {args.parameters}...")
    with open(args.parameters, "r") as f:
        params = json.load(f)

    print("Creating experiments list...")
    experiments = create_experiments_list(params)

    if args.experiment is not None:
        print(f"Saving experiments list to {args.experiment}...")
        save_experiments_list(args.experiment, experiments)
    else:
        print(json.dumps(experiments, indent=4))
    if args.script is not None:
        print(f"Saving slurm script to {args.script}...")
        save_slurm_script(args.script, len(experiments), args.experiment)


def run(args):
    """Main function to run experiments"""

    print(f"Loading experiments from {args.experiment}...")
    with open(args.experiment, "r") as f:
        experiments = json.load(f)

    item = experiments[args.index]
    item["index"] = args.index 

    print(f"Running experiment {args.index + 1}/{len(experiments)}: {item}")
    # TODO: Edit here the code to run your experiment
    process(**item)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        prog="run experiment",
        description="Generate experiment list and slurm script and run experiments.",
    )

    subparser = parser.add_subparsers()

    # first command
    parser_cnf = subparser.add_parser("configure")
    parser_cnf.add_argument(
        "--parameters", required=True, type=Path, help="Path to parameters JSON file."
    )
    parser_cnf.add_argument(
        "--experiment", type=Path, help="Path to save experiments list JSON file."
    )
    parser_cnf.add_argument(
        "--script", type=Path, help="Path to save slurm script file."
    )
    parser_cnf.set_defaults(func=configure)

    # second command
    parser_run = subparser.add_parser("run")
    parser_run.add_argument(
        "--experiment",
        type=Path,
        help="Path to experiments list JSON file.",
        required=True,
    )
    parser_run.add_argument(
        "--index",
        type=int,
        help="Index of the experiment to run.",
        required=True,
    )
    parser_run.set_defaults(func=run)
    args = parser.parse_args()

    args.func(args)