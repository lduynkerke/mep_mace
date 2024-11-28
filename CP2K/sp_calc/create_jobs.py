import os
import subprocess

def create_slurm_script(folder, start, end):
    script = f"""#!/bin/bash
        #SBATCH --array={start}-{end}%8
        #SBATCH --job-name=sp_{start}_{end}
        #SBATCH --cpus-per-task=48
        #SBATCH --time=2:00:00
        #SBATCH --output={folder}/output/output_%A_%a.out  # Output file for each job
        #SBATCH --partition=genoa
        
        module load 2024 CP2K/2024.3-foss-2024a

        echo "{folder}/conf_${{SLURM_ARRAY_TASK_ID}}"
        cd "{folder}/conf_${{SLURM_ARRAY_TASK_ID}}"
        
        srun --exclusive cp2k.popt -i sp_calc_lda.inp -o lda.out
        srun --exclusive cp2k.popt -i sp_calc_blyp.inp -o blyp.out
        srun --exclusive cp2k.popt -i sp_calc_pbe0.inp -o pbe0.out
        """
    return script

def submit_jobs(root_dir):
    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "coords" in dirnames:
            coords_dir = os.path.join(dirpath, "coords")

            for start, end in [(0, 49)]:
                job_script = create_slurm_script(coords_dir, start, end)
                script_filename = f"array_job_{start}_{end}.sh"
                script_path = os.path.join(coords_dir, script_filename)
                print(f"{script_path} submitted")
                with open(script_path, 'w') as f:
                    f.write(job_script)

                submit_command = f"sbatch {script_path}"
                subprocess.run(submit_command, shell=True, check=True)

if __name__ == "__main__":
    folder = '../data/coords'
    #submit_jobs(folder)
    create_slurm_script(folder, 0, 602)