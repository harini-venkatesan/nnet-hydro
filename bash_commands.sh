#!/bin/bash -l

#SBATCH --nodes=1 # Allocate *at least* 5 nodes to this job.
#SBATCH --ntasks-per-node=1 # Allocate *at most* 5 tasks for job steps in the job
#SBATCH --cpus-per-task=10 # Each task needs only one CPU
#SBATCH --mem=10G # This particular job won't need much memory
#SBATCH --time=6-00:0:00  # 1 day and 1 minute 
#SBATCH --mail-type=ALL
#SBATCH --job-name="hydro"
#SBATCH -p batch # You could pick other partitions for other jobs
#SBATCH --wait-all-nodes=1  # Run once all resources are available
#SBATCH --output=outputs/output_%j-%N.txt # logging per job and per host in the current directory. Both stdout and stderr are logged.

# Place any commands you want to run below
#singularity exec /data/ChristianShelton/containers/rlairgpu1-1.sif bash sing_commands.sh

#base_compiledir=/var/tmp/.theano
conda activate fenics-torch
#cd adaptive-rho-mcmc
python3 -u main.py

