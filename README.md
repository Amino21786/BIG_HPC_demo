# Template HPC repository to run large-scale simulations - BIG meeting - 14/01/2026
## Setting up on Nimbus for use of CPU and GPU instances in Python
Aim for everyone to be set up on Nimbus and be able to run Python scripts (e.g. Optimisation algorithm simulation over many iterations for an inverse problem)


Files:
- process.py - where the main code for the simulation exists (the core code that you would usually run on your own machine)
- run_experiment.py - where the SLURM script (.sh) is created for the process.py to run the simulation (sets up all the features to run on the HPC cluster)

Folders all named for your use.


# Steps to get into the right HPC cluster:

## On Command line (terminal/powershell etc):
1) Sign into Nimbus on the command line with 'ssh bath_username@nimbus.hpc.ac.uk'
2) Navigate to the right directory 'cd /campaign/EB-MA3194M-019/BIG_HPC_demo'

## IDE Editor (VSCode)
1) Click on bottom left with the blue icon and "Connect to host" -> "Add new SSH host" -> enter bath_username@nimbus.hpc.ac.uk and fill in your password
2) Set up vscode-server with extensions - Python is the only one needed
3) Open folder with the directory of '/campaign/EB-MA3194M-019/BIG_HPC_demo







## References
Based off https://github.com/jboulanger/template_slurm_experiment repository to produce experiment scripts to run on a HPC cluster.

  
