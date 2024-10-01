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
- **CP2K**: Folder containing scripts and input/output files of CP2K runs.
  - **templates**: Folder containing CP2K input templates.
    - `ab_initio_MD_templ.inp`: input file template for aiMD CP2K runs.
    - `geometry_optimization_templ.inp` input file template for geometry optimization CP2K runs.
    - `velocity_softening_dynamics`: input file template for velocity softening dynamics CP2K runs.
  - `CP2K.sh`: bash script that starts CP2K simulations using a given input file.
  - `single_point_LDA.inp`: input file for running LDA single point calculations using Sasha's suggestions.
  - `single_point_LDA_trajectory.inp`: input file with an attempt to read file as trajectory - fails.
  - `single_point_lda_av_353K_verbose.out`: output file for `single_point_LDA.inp`.
- **MACE**:
- **Results**: Folder containing results, similarly structured to data folder.