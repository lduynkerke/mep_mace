#!/bin/sh
#SBATCH -t 3:30:00
#SBATCH -p compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --account=Education-AS-MSc-AP
#SBATCH -J cp2k_test
#SBATCH --mem-per-cpu=3G
 
module load 2024r1
module load openmpi/4.1.6
module load cp2k/2023.2

cp2k.popt -i sp_calc_pbe.inp -o pbe.out

