# mep_mace
This repo contains code for my MEP.

## Contents
- **ASE**: Folder containing scripts to process energy and forces using ASE:
  - process_energy_mpi.py: parallelized script to load structures and calculate structure energy.
  - submit_slurm_jobs.py: script to scan a data folder and create slurm jobs for each XYZ file.
  - merge_csvs.py: script to merge and rename CSV files within the same directory into one single CSV.
  - plot_folder.py: script to gather all CSVs from a data folder and create matplotlib PNG plots.

- **MACE**:
- **Results**: Folder containing results, similarly structured to data folder.