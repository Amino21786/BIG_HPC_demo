from pathlib import Path
#from tqdm import tqdm
import json
import scipy
import numpy as np

from skimage.data import shepp_logan_phantom
from skimage.transform import resize

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

def gradient_descent(noisy_image, original_image, stepsize, max_iter):
    """Simple gradient descent for image denoising."""
    # Initialize the denoised image
    denoised_image = noisy_image.copy()
    rel_errors = []
    for i in range(max_iter): #tqdm(range(max_iter), desc="Gradient Descent"):
        # Compute the gradient (identity operator with noise
        gradient = denoised_image - noisy_image
        # Update the denoised image
        denoised_image -= stepsize * gradient
        # Compute relative error
        rel_error = np.linalg.norm(denoised_image - original_image) / np.linalg.norm(original_image)
        rel_errors.append(rel_error)
        print(f"Iteration {i+1}/{max_iter}, Relative Error: {rel_error:.6f}")

    return denoised_image, rel_errors


def process(**kwargs):
   

    # Experiment index for naming
    exp_id = kwargs.get("index", 0)
    experiment_name = kwargs.get("name", "experiment")
    algorithm_name = kwargs.get("algorithm", "gradient_descent")
    stepsize = kwargs.get("stepsize", 1e-2)
    max_iter = kwargs.get("max_iter", 10)

    # Creating output directory:
    outdir = Path(f"results/{experiment_name}/{algorithm_name}/exp_{exp_id:03d}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {outdir}")

    # -------------------------------------------
    # Running of experiment for image denoising of the Shepp-Logan phantom
    phantom = shepp_logan_phantom()
    image = resize(phantom, (128, 128), mode='reflect', anti_aliasing=True)
    noisy_image = image + 0.1 * np.random.randn(*image.shape)
    

    print("Running the denoising algorithm...")
    if algorithm_name == "gradient_descent":
        denoised_image, rel_errors = gradient_descent(
            noisy_image,
            image,
            stepsize,
            max_iter
        )
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")
    
    print("Denoising  algorithm completed.")
   
    # -------------------------------------------
    # SAVE results as a numpy compressed file
    # -------------------------------------------
    print("Saving results into a single compressed file...")
    results_path = outdir / f"{algorithm_name}_results.npz"

    np.savez_compressed(
        results_path,
        results=denoised_image,
        rel_errors=rel_errors
        #dual=torch.stack(dual_rec).detach().cpu().numpy()
    )


    """
    print("Saving info dictionary...")
    info_np = to_jsonable(info)
    np.savez_compressed(
        outdir / f"info_{initialisations_choice}_gamma_{gamma:.1f}_alpha_{alpha}.npz",
        info=np.array(info_np, dtype=object)
    )
    """
   
 

    print("Experiment done.")





