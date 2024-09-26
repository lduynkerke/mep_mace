#!/bin/bash
#SBATCH --job-name="run_ase"
#SBATCH --partition=compute
#SBATCH --time=3:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --account=Education-AS-MSc-AP

module load 2024r1
module load python
module load py-numpy
module load py-scipy
module load py-pandas
module load py-matplotlib
module load py-pip
module load py-torch/2.1.0

srun python submit_slurm_jobs.py
