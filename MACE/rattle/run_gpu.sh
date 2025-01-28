#!/bin/bash
#SBATCH --job-name="mace_gpu"
#SBATCH --partition=gpu
#SBATCH --time=9:59:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --gpus-per-task=1
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-AS-MSc-AP

module load 2024r1
module load miniconda3
module load cuda/11.7

conda activate pytorch20_env

dir="../rattled"
mace_dir="mace50"
iters=50

mkdir -p "$mace_dir"; cd "$mace_dir"

echo "Merging CP2K files and retraining MACE..."
python ../cp2k_to_xyz.py --root="$dir" --base='conf_' --range 0 $iters 1\
    --atomic_numbers='H:1,O:8,Si:14,C:6,Zr:40' --lattice="21.5 0.0 0.0 0.0 21.5 0.0 0.0 0.0 40.0" || { echo "Error: cp2k_to_xyz.py failed."; exit 1; }

if [[ ! -f "../training.xyz" ]]; then
    echo "Error: training.xyz not found..."
    exit 1
fi

cat "../training.xyz" "$dir/merged.xyz" > training_rattled.xyz || { echo "Error: Failed to merge XYZ files."; exit 1; }

mace_run_train --name="rattled50" --train_file="training_rattled.xyz" \
    --valid_fraction=0.33 --foundation_model="medium" \
    --max_num_epochs=100 --device=cuda --batch_size=2 \
    --valid_batch_size=1 --seed="50" --forces_key="force" \
    --energy_key="TotEnergy" --restart_latest \
    --E0s='{1: -0.46400, 6: -5.30392, 8: -15.75010, 14: -3.71436, 40: -46.84768}' || { echo "Error: MACE training failed."; exit 1; }

echo "Running molecular dynamics..."
python ../run_md.py --model="../mace.model" --structure="$dir/conf_0/original.xyz" \
    --temp=353 --T0=100 --niter=$iters --traj_name="out.xyz" --logfile="md.log" || { echo "Error: MD simulation failed."; exit 1; }

conda deactivate
