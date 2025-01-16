#!/bin/bash
#SBATCH --time=23:59:00
#SBATCH --partition=gpu
#SBATCH --ntasks=1
#SBATCH --gpus-per-task=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=5G
#SBATCH --account=Education-AS-MSc-AP
#SBATCH --job-name=iter_mace

if [ -z "$1" ]; then
    echo "Usage: $0 <iteration_index>"
    exit 1
fi

module load 2024r1
module load miniconda3
module load cuda/11.7

conda activate pytorch20_env

mpiexec --version

i=$1
dir="conf_$i"
prev_dir="conf_$((i-1))"
FILE1="sp_calc_pbe.inp"
FILE2="run_cp2k.sh"
job_ids=()

mkdir -p "$dir"; cd "$dir"

# Step 1: Prepare files from previous iteration, only if i > 0
echo "Merging CP2K files for iteration $i..."
if [[ $i -gt 0 ]]; then
    if [[ ! -f "../$prev_dir/training.xyz" ]]; then
        echo "Error: training.xyz not found in $prev_dir."
        exit 1
    fi

    if [[ ! -f "../$prev_dir/mace.model" ]]; then
        echo "Error: mace.model not found in $prev_dir."
        exit 1
    fi
    
    python ../cp2k_to_xyz.py --root="../$prev_dir" --base='coord_' --range 0 5 1 \
        --atom_count=506 --atomic_numbers='H:1,O:8,Si:14,C:6,Zr:40' \
        --lattice="21.5 0.0 0.0 0.0 21.5 0.0 0.0 0.0 40.0" || { echo "Error: cp2k_to_xyz.py failed."; exit 1; }

    cat "../$prev_dir/training.xyz" "../$prev_dir/merged.xyz" > training.xyz || { echo "Error: Failed to merge XYZ files."; exit 1; }
    mkdir -p checkpoints; cp -r "../$prev_dir/checkpoints" .
    cp "../$prev_dir/coord_5/pbe.xyz" ./struct.xyz
fi

# Step 2: Retrain MACE
for j in {0..4}
do
    echo "Training MACE model $j..."
    mace_run_train --name="mace_$j" --train_file="training_$j.xyz" \
        --valid_fraction=0.33 --foundation_model="medium" \
        --max_num_epochs=$((5*i+5)) --device=cuda --batch_size=2 \
        --valid_batch_size=1 --seed="$j" --forces_key="force" \
        --energy_key="TotEnergy" --restart_latest \
        --E0s='{1: -0.46400, 6: -5.30392, 8: -15.75010, 14: -3.71436, 40: -46.84768}' || { echo "Error: MACE training failed."; exit 1; }
done

# Step 3: Run molecular dynamics
echo "Running molecular dynamics for iteration $i..."
python ../run_md_std.py --models "mace_0.model" "mace_1.model" "mace_2.model" "mace_3.model" "mace_4.model" \
    --structure="struct.xyz" --temp=353 --T0=100 --traj_name="out.xyz" --logfile="md.log" \
    --niter=6 --nmax=10000 --std_threshold=0.005 || { echo "Error: MD simulation failed."; exit 1; }

conda deactivate

# Step 4: Extract and save last frames from MD trajectory
echo "Extracting coordinates for CP2K force calculations..."
python ../extract_coords.py -i "out.xyz" -o "." -n 6 || {
    echo "Error: Coordinate extraction failed."
    exit 1
}

# Step 5: Run CP2K on extracted frames
echo "Running CP2K force calculations..."
for subdir in ./coord*/; do
    if [ -d "$subdir" ]; then
        cp "../$FILE1" "$subdir"
        cp "../$FILE2" "$subdir"
        echo "Copied $FILE1 and $FILE2 to $subdir"

        cd "$subdir" || exit
        job_id=$(sbatch --no-requeue "$FILE2" | grep -o '[0-9]\+')  # Capture job ID
        job_ids+=("$job_id")
        echo "Submitted CP2K job in $subdir with ID: $job_id"
	cd - > /dev/null || exit
    fi
done

# Step 6: Wait for CP2K jobs to finish, then start next iteration
if [[ ${#job_ids[@]} -eq 0 ]]; then
    echo "Error: No CP2K jobs were submitted."
    exit 1
fi

dependency_string=$(IFS=:; echo "${job_ids[*]}")
echo "Submitting GPU job run.sh for iteration $((i+1)) with dependencies on jobs: $dependency_string..."
cd ..
sbatch --dependency=afterok:$dependency_string run.sh $((i+1)) || {
    echo "Error: Failed to submit next GPU job."
    exit 1
}

conda deactivate
