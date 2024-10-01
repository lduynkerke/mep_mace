#!/bin/sh
#SBATCH -t 3:30:00
#SBATCH -p compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=48
#SBATCH --account=Education-AS-MSc-AP
#SBATCH -J cp2k_test
#SBATCH --mem=120G

module load 2024r1
module load openmpi/4.1.6
module load cp2k/2023.2

srun cp2k.popt -i single_point_LDA.inp -o single_point_lda_av_353K.out
