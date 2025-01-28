#!/bin/sh
#SBATCH -t 3:30:00
#SBATCH -p compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem-per-cpu=3G
#SBATCH --account=Education-AS-MSc-AP
#SBATCH -J rattle

module load 2024r1
module load openmpi/4.1.6
module load cp2k/2023.2
module load python
module load py-numpy
module load py-scipy

dir="rattle100"
FILE1="sp_calc_pbe.inp"
FILE2="run_cp2k.sh"
subdirs=()

# rattle
python rattle.py -i training.xyz -o $dir -n 100 -r 0.25

# Find subdirectories
for subdir in "$dir"/*; do
    if [ -d "$subdir" ]; then
        subdirs+=("$subdir")
    fi
done

if [ ${#subdirs[@]} -eq 0 ]; then
    echo "No subdirectories found in $dir. Exiting."
    exit 1
fi

if [ -z "$FILE1" ] || [ -z "$FILE2" ]; then
    echo "FILE1 or FILE2 is not set. Exiting."
    rm "$subdir_file"
    exit 1
fi

# Create job script
job_script=$(mktemp)
cat <<EOF > "$job_script"
#!/bin/bash
#SBATCH -t 3:30:00
#SBATCH -p compute
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --account=Education-AS-MSc-AP
#SBATCH -J cp2k_array
#SBATCH --mem-per-cpu=3G

module load 2024r1
module load openmpi/4.1.6
module load cp2k/2023.2

# Define the subdirectory for this task
subdirs=($(
    find "$dir" -mindepth 1 -maxdepth 1 -type d
))
subdir="\${subdirs[\$SLURM_ARRAY_TASK_ID]}"

#subdir=$(realpath "$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" "$1")")
#subdir=\$(sed -n "\$((SLURM_ARRAY_TASK_ID + 1))p" \$1)

# Copy files into the subdirectory
cp "$FILE1" "\$subdir" || { echo "Failed to copy $FILE1 to \$subdir"; exit 1; }
cp "$FILE2" "\$subdir" || { echo "Failed to copy $FILE2 to \$subdir"; exit 1; }

# Run CP2K in the subdirectory
cd "\$subdir" || { echo "Failed to enter \$subdir"; exit 1; }
cp2k.popt -i sp_calc_pbe.inp -o pbe.out || { echo "CP2K failed in \$subdir"; exit 1; }
cd - > /dev/null || exit
EOF

# Submit job script
subdir_file=$(mktemp)
printf "%s\n" "${subdirs[@]}" > "$subdir_file"

# Debug
echo "Job script content:"
cat "$job_script"
echo "Subdirectory file content:"
cat "$subdir_file"

array_job_id=$(sbatch --array=0-$((${#subdirs[@]} - 1)) "$job_script" "$subdir_file" | grep -o '[0-9]\+')
echo "Array job submitted with ID: $array_job_id"

# Submit GPU job with dependency
echo "Submitting GPU job to train MACE with dependencies on jobs: $array_job_id..."
sbatch --dependency=afterok:$array_job_id run_gpu.sh || {
    echo "Error: Failed to submit GPU job."
    exit 1
}
