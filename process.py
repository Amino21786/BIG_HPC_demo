import os
import json
import torch
import numpy as np
from pathlib import Path
from datetime import datetime


def to_jsonable(obj):
    """Convert nested tensors/arrays to JSON-serializable structures."""
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().numpy()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {k: to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    return obj

def initialisations(choice = "zeros-zeros", data_input=None, problem=None):
    """Function to create initialisations based on choice string."""
    hterms = [t for t in problem if t.type == "h"]
    
    if choice == "zeros-zeros":
        x0 = 0*torch.stack([data_input, data_input])
        y0 = [0*term.A(x0) for term in hterms]
        return x0, y0
    elif choice == "data-zeros":
        x0 = torch.stack([data_input, data_input])
        y0 = [0*term.A(x0) for term in hterms]
        return x0, y0

def discrepancy_principle(data, alpha_info, noise='mixed'):
    """
    Function to compute the discrepancies for Gaussian and Poisson terms
    Args:
        data: measured data tensor
        info: info dictionary from optimization

    Returns:
        gauss_disc: float
        poisson_disc: float
    """
    
    n_pixels = data.numel()
    # print the different components of Cost from info
    print(f"Cost components: {alpha_info['Cost'][0]}")
    # last cost values
    print(f"Final Cost components: {alpha_info['Cost'][-1]}")
    #return True
    if noise == 'mixed':
        final_gaussian_cost = alpha_info['Cost'][-1][1]  # Assuming Gaussian term is the second term
        final_poisson_cost = alpha_info['Cost'][-1][2]  # Assuming Poisson term is the third term
    elif noise == 'gaussian':
        final_gaussian_cost = alpha_info['Cost'][-1][1]
        final_poisson_cost = 0.0
    elif noise == 'poisson':
        final_gaussian_cost = 0.0
        final_poisson_cost = alpha_info['Cost'][-1][1]

    gaussian_discrepancy = final_gaussian_cost / (n_pixels)
    poisson_discrepancy = final_poisson_cost / (n_pixels)
    print(f"Final Gaussian Cost: {final_gaussian_cost:.4f}, Gaussian Discrepancy: {gaussian_discrepancy:.4f}")
    print(f"Final Poisson Cost: {final_poisson_cost:.4f}, Poisson Discrepancy: {poisson_discrepancy:.4f}")

    return gaussian_discrepancy.item(), poisson_discrepancy.item()

def process(**kwargs):
    # -------------------------------------------
    # Setup device
    # -------------------------------------------
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Running on device: {device}")

    # Experiment index for naming
    exp_id = kwargs.get("index", 0)

    algorithm_name = kwargs.get("algorithm", "pd3o-alphas") 

    # Create output directory: results/<algorithm>/exp_XXX/
    outdir = Path(f"results/{algorithm_name}/exp_{exp_id:03d}")
    outdir.mkdir(parents=True, exist_ok=True)
    print(f"Saving outputs to: {outdir}")

    # -------------------------------------------
    # Print experiment parameters
    # -------------------------------------------
    print("Parameters:")
    for k, v in kwargs.items():
        print(f"  {k}: {v}")

    # Taken from the JSON parameters (algorithm parameters)
    # starting alpha and gamma (2e-4, 100)
    alpha_ref = 2e-4
    gamma_ref = kwargs.get("gamma_ref", 100)
    alpha = kwargs.get("alpha", 2e-4)
    max_iter = kwargs.get("max_iter", 10)
    initialisations_choice = kwargs.get("initialisation", "zeros-zeros")
    tau = kwargs.get("tau", 10)

    gamma = gamma_ref * (alpha_ref / alpha)
    sigma = tau / gamma
    
    print(f"PD3O stepsizes for alpha={alpha}: tau={tau}, sigma={sigma}, gamma={gamma}")
    
    """
    # Algorithm parameters for each experiment
    alpha = kwargs.get("alpha", 2e-4)
    gamma = gamma_ref * (alpha_ref / alpha)**2
    rho = kwargs.get("rho", 1)
    max_iter = kwargs.get("max_iter", 10)
    initialisations_choice = kwargs.get("initialisation", "zeros-zeros")
    c = 1 #1/op_norm(M) #M the forward operator for Widefield is 1
    tau = np.sqrt(gamma*rho)*c 
    sigma = c*np.sqrt(rho/gamma) #c/sqrt(gamma*rho)
    print(f"Gamma: {gamma}, Alpha: {alpha}, Tau: {tau}, Sigma: {sigma}")
    """
  

    # -------------------------------------------
    # Load data and setup operators
    # -------------------------------------------
    from pylsdeconv.generation import generate_fibers, generate_sensor_noise
    from pylsdeconv.microscopy import Widefield
    from pylsdeconv.optimization import algorithm, cost, metrics, operator, function
    from pylsdeconv.pnp import unet, denoiser
    
    model = unet.load_model("models/tmp.pth").to(device)
  
    sigman = 30

    # values for the first set of ground truth and measurement
    #shape = [16, 128, 128]
    #spacing = [200, 100, 100]
    #readout = 5
    
    shape = [32, 96, 96]
    spacing = [100, 65, 65]
    readout = 1.7 # Gaussian readout noise level from the electronic circuit

    # Define forward microscopy operator and load data
    op = Widefield(shape, spacing, 500, 1.0, 1.3, device=device)
    # op = LightSheet(shape, spacing, 500, 520, 0.75, 1.2, 1.33, device=device)
    
    img = torch.load("data/ground_truth_setting.pt").to(device)
    data = torch.load("data/measurement_setting.pt").to(device)
    #data = torch.load("data/fibres_lsmmeasurement.pt").to(device)
    # data = generate_sensor_noise(op(img), 1, 0, 5) # new poisson and gaussian noise added each time


    # -------------------------------------------
    # Define cost function
    # -------------------------------------------
    J = [
       cost.LSRTerm(
        denoiser.ResDen(model, sigman),
        alpha=alpha,
        op = operator.Stack([[operator.Identity(), operator.Null()]]),
        type="f",   
    ),
        cost.Term(
            fun=function.SSEStackNonNegative(factor=0.5 / (readout**2), data=data),
            type="g",
        ),
        cost.Term(
            function.KLD2(),
            operator.Stack([[op, operator.Null()],
                            [operator.Null(), operator.Identity()]]),
            "h",
        ),
    ]

    perf = {
        "MSE": metrics.MSE(
            img, op=operator.Stack([[operator.Identity(), operator.Null()]])
        ),
        "RelMSE": metrics.RelMSE(
            img, op=operator.Stack([[operator.Identity(), operator.Null()]])
        ),
        "DKL0": metrics.KLD(data, operator.Stack([[op, operator.Null()]]), 0.1),
        "Cost": metrics.Cost(J),
        "Time": metrics.Chronograph(),
        #"Iterates": ,
    }

    # Setup of parameters and initialisation for primal and dual variables
    params = { "tau": tau,  "max_iter": max_iter, "sigma": sigma}
    x0, y0 = initialisations(initialisations_choice, data_input=data, problem=J)

    # -------------------------------------------
    # Run PD3O algorithm
    # -------------------------------------------
    rec, _, info = algorithm.pd3o(J, x0, y0, **params, metrics=perf)
    # -------------------------------------------
    # Print final metrics
    # -------------------------------------------

    # print all the info metrics apart from 'iterates'
    for key, values in info.items():
        if key != "iterates":
            print(f"  {key}: {values[-1]}")

    # discrepancy principle value at final iteration
    gaussian_disc, poisson_disc = discrepancy_principle(data, info, noise='mixed')
    discrepancies = np.array([gaussian_disc, poisson_disc])

    # -------------------------------------------
    # SAVE results (rec, dual_rec, info, params, discrepancies) as a numpy compressed file
    # -------------------------------------------
    #gamma_rounded to 2 significant figures
    print("Saving rec + dual_rec into a single compressed file...")
    rec_dual_path = outdir / f"rec_dual_{initialisations_choice}_gamma_{gamma:.1f}_alpha_{alpha}.npz"

    np.savez_compressed(
        rec_dual_path,
        rec=rec.detach().cpu().numpy()
        #dual=torch.stack(dual_rec).detach().cpu().numpy()
    )

    print("Saving info dictionary...")
    info_np = to_jsonable(info)
    np.savez_compressed(
        outdir / f"info_{initialisations_choice}_gamma_{gamma:.1f}_alpha_{alpha}.npz",
        info=np.array(info_np, dtype=object)
    )

    print("Saving parameter dictionary...")
    with open(outdir / "params.json", "w") as f:
        json.dump(to_jsonable(kwargs), f, indent=4)

    print("Saving discrepancies...")
    np.savez_compressed(
        outdir / f"discrepancies_{initialisations_choice}_gamma_{gamma:.1f}_alpha_{alpha}.npz",
        discrepancies=discrepancies
    )

    print("Done.")





