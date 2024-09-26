# mep_mace
This repo contains code for my MEP.

## Contents
- **ASE**: Folder containing scripts to process energy and forces using ASE:
  - `submit_jobs.sh`: bash script that calls `submit_slurm_jobs.py` to create jobs.
  - `postprocess.sh`: bash script that calls `merge_csvs.py` to rename and merge csv-files, making them ready to be plotted. Next, it calls plot_folder.py to generate plots.
  - `process_energy_mpi.py`: parallelized script to load structures and calculate structure energy.
  - `submit_slurm_jobs.py`: script to scan a data folder and create slurm jobs for each XYZ file.
  - `merge_csvs.py`: script to merge and rename CSV files within the same directory into one single CSV.
  - `plot_folder.py`: script to gather all CSVs from a data folder and create matplotlib PNG plots.

- **MACE**:
- **Results**: Folder containing results, similarly structured to data folder.