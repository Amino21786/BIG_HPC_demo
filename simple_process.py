from pathlib import Path
#from tqdm import tqdm
import numpy as np


"""
def to_jsonable(obj):
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj
"""

def process(**kwargs):
   

    # Experiment index for naming
    exp_id = kwargs.get("index", 0)
    name = kwargs.get("name", "experiment")
    experiment_name = "matrix_vector" #kwargs.get("name", "experiment")
    

    # Creating output directory:
    outdir = Path(f"results/{name}/{experiment_name}/exp_{exp_id:03d}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {outdir}")

    # -------------------------------------------
    # Running of experiment of doing numpy matrix-vector multiplication

    A = np.array([[1, 2], [3, 4]])
    v = np.array([5, 6])

    result = np.matmul(A, v)
    print(result)

    print("printing norm of result:")
    print(np.linalg.norm(result))
 

    print("Saving results into a single compressed file...")
    results_path = outdir / f"{experiment_name}_results.npz"

    np.savez_compressed(
        results_path,
        results=result
    )

    print("Experiment done.")





