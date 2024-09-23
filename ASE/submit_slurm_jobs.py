import os
import subprocess

def submit_jobs(data_folder, results_folder, model_path, script_path):
    """
    Submits SLURM jobs to process XYZ files in parallel using MPI.

    This function traverses through the `data_folder`, identifies `.xyz` files, creates corresponding
    result directories in the `results_folder`, and generates SLURM job submission scripts to run
    an energy calculation using a specified model. Each job is submitted using the `sbatch` command.

    :param data_folder: The base directory where input `.xyz` files are located.
    :type data_folder: str
    :param results_folder: The base directory where results will be saved.
    :type results_folder: str
    :param model_path: Path to the MACE model file used for energy calculations.
    :type model_path: str
    :param script_path: Path to the MPI script that processes energy from the XYZ files.
    :type script_path: str
    :return: None
    """
    for root, dirs, files in os.walk(data_folder):
        xyz_files = [f for f in files if f.startswith('Forces_and_traj') and f.endswith('.xyz')]
        print(xyz_files)

        if not xyz_files:
            continue

        # Process each xyz file separately
        for xyz_file in xyz_files:
            input_file = os.path.join(root, xyz_file)
            relative_path = os.path.relpath(root, data_folder)
            results_subfolder = os.path.join(results_folder, relative_path)

            os.makedirs(results_subfolder, exist_ok=True)
            output_csv = os.path.join(results_subfolder, f"{os.path.splitext(xyz_file)[0]}.csv")

            # Prepare the command to call process_energy.py for this file
            command = [
                "srun python", script_path,
                input_file,  # Single xyz file
                "--model_path", model_path,
                "--output_csv", output_csv
            ]

            # Write the SLURM job submission script for this task
            job_script = f"""#!/bin/bash
                            #SBATCH --job-name='mpi_energy'
                            #SBATCH --partition=compute
                            #SBATCH --time=4:00:00  # Adjust time limit as needed
                            #SBATCH --ntasks=36
                            #SBATCH --cpus-per-task=1  # Adjust resources as needed
                            #SBATCH --mem=185G
                            #SBATCH --account=Education-AS-Msc-AP
                            
                            module load 2024r1
                            module load openmpi
                            module load python  # Load any required modules
                            module load py-mpi4py
                            module load py-numpy
                            module load py-scipy
                            module load py-pandas
                            module load py-matplotlib
                            module load py-pip
                            { ' '.join(command) }
                        """

            job_file = os.path.join(results_subfolder, "submit_job.sh")
            with open(job_file, "w") as f:
                f.write(job_script)

            # Submit the job script using sbatch
            subprocess.run(["sbatch", job_file])

    print("All jobs submitted.")

def main():
    """
    The main function that triggers the submission of SLURM jobs for processing XYZ files.

    This function uses predefined paths for data, results, model, and script files to
    process energy calculations in a distributed computing environment using SLURM.

    :return: None
    """
    data_folder = "../../../data/"
    results_folder = "../../../results/"
    model_path = "../../../../mace/mace/calculators/foundations_models/2023-12-03-mace-mp.model"
    script_path = "../process_energy_mpi.py"

    submit_jobs(data_folder, results_folder, model_path, script_path)


if __name__ == "__main__":
    main()