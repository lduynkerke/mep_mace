import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

def load_npz_data(base_path, strain_surface, sim_type, temp_folder, folder=None, file_name='forces_data.npz'):
    """
    Loads npz data for a specific strain surface, simulation type, and temperature folder.

    Constructs the path to an npz file using the specified base path, strain surface,
    simulation type, temperature folder, and optionally a subfolder. Loads the data
    if the file exists, returning None if the file does not exist or an error occurs.

    :param base_path: The root directory containing the data.
    :type base_path: str
    :param strain_surface: The strain surface folder name.
    :type strain_surface: str
    :param sim_type: The simulation type (e.g., 'Adsorption_Complex_Simulations').
    :type sim_type: str
    :param temp_folder: The temperature folder (e.g., 'aiMD_353K').
    :type temp_folder: str
    :param folder: Optional subfolder within the temp_folder.
    :type folder: str, optional
    :param file_name: The npz file name to load (default is 'forces_data.npz').
    :type file_name: str
    :return: Loaded data if successful; otherwise, None.
    :rtype: numpy.lib.npyio.NpzFile or None
    """
    if folder is None:
        npz_path = os.path.join(base_path, strain_surface, sim_type, temp_folder, file_name)
    else:
        npz_path = os.path.join(base_path, strain_surface, sim_type, temp_folder, folder, file_name)

    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            return data
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            return None
    else:
        print(f"File {npz_path} does not exist.")
        return None

def plot_energies(base_path, sim_types, sim_type_legends, strain_surfaces, strain_surface_names, temperature_folders,
                  temp_folder_names, axis_lims=None):
    """
        Plots energy comparisons across different strain surfaces, temperatures, and simulation types.

        Generates scatter plots comparing Perdew-Burke-Ernzerhof (PBE) and Local Density Approximation (LDA)
        energies per atom for each strain surface and temperature, including a regression line and RMSE.
        Saves each plot for each simulation type.

        :param base_path: Base directory for saving plots.
        :type base_path: str
        :param sim_types: List of simulation types to compare.
        :type sim_types: list[str]
        :param sim_type_legends: Legends for each simulation type for plot titles.
        :type sim_type_legends: list[str]
        :param strain_surfaces: Strain surface identifiers.
        :type strain_surfaces: list[str]
        :param strain_surface_names: Names of strain surfaces for display.
        :type strain_surface_names: list[str]
        :param temperature_folders: Temperature folder identifiers.
        :type temperature_folders: list[str]
        :param temp_folder_names: Names of temperature folders for display.
        :type temp_folder_names: list[str]
        :param axis_lims: Axis limits for each simulation type's plot.
        :type axis_lims: list[list[tuple]], optional
        :return: None
        """
    for sim_idx, sim_type in enumerate(sim_types):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for i, strain_surface in enumerate(strain_surfaces):
            for j, temp_folder in enumerate(temperature_folders):
                data = load_npz_data(base_path, strain_surface, sim_type, temp_folder, folder='coords')
                mace_energies, _ = load_npz_data(base_path, strain_surface, sim_type, temp_folder, file_name='mace1_forces.npz')

                ax = axes[i, j]
                if data is not None:
                    # Energies are stored as size (50)
                    pbe_energies = data['pbe_energies']
                    lda_energies = data['lda_energies']
                    gga_energies = data['gga_energies']


                    atoms = data['pbe_forces'].shape[1]  # Assuming forces are of shape (50, N, 3)
                    pbe_energies_per_atom = pbe_energies / atoms * 27.2114079527
                    lda_energies_per_atom = lda_energies / atoms * 27.2114079527

                    ax.scatter(pbe_energies_per_atom, lda_energies_per_atom, label=f'LDA Energies')

                    rmse = np.sqrt(np.mean((pbe_energies_per_atom - lda_energies_per_atom) ** 2))
                    ax.text(0.95, 0.10, f'RMSE = {rmse:.4f}', ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

                    slope, intercept, _, _, _ = scipy.stats.linregress(pbe_energies_per_atom, lda_energies_per_atom)
                    sns.regplot(x=pbe_energies_per_atom, y=lda_energies_per_atom, ax=ax, scatter=False, color='r')
                    ax.text(0.95, 0.05, f'y = {intercept:.3f} + {slope:.3f}x', ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

                    strain_surface_name = strain_surface_names[i] if i < len(strain_surface_names) else strain_surface
                    temp_folder_name = temp_folder_names[j] if j < len(temp_folder_names) else temp_folder
                    ax.set_title(f'{strain_surface_name} - {temp_folder_name}')

                    if i == 2:
                        ax.set_xlabel('PBE Energy / atom (eV)')
                    if j == 0:
                        ax.set_ylabel('LDA Energy / atom (eV)')

                    if axis_lims is not None:
                        ax.set_xlim(axis_lims[sim_idx][0])
                        ax.set_ylim(axis_lims[sim_idx][1])
                        ax.set_xticks(axis_lims[sim_idx][2])
                        ax.set_yticks(axis_lims[sim_idx][3])
                else:
                    ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
                    ax.set_title(f'{strain_surface_name} - {temp_folder_name}')

        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        fig.savefig(os.path.join(base_path, f'Energy_Comparison_{sim_type}_lda.png'))
        plt.show()

def plot_forces(base_path, sim_types, sim_type_legends, strain_surfaces, strain_surface_names, temperature_folders,
                temp_folder_names, plot_both=False):
    """
        Plots force comparisons across different strain surfaces, temperatures, and simulation types.

        Generates scatter plots comparing Perdew-Burke-Ernzerhof (PBE) forces to LDA and/or Generalized
        Gradient Approximation (GGA) forces for each strain surface and temperature. Saves each plot
        for each simulation type.

        :param base_path: Base directory for saving plots.
        :type base_path: str
        :param sim_types: List of simulation types to compare.
        :type sim_types: list[str]
        :param sim_type_legends: Legends for each simulation type for plot titles.
        :type sim_type_legends: list[str]
        :param strain_surfaces: Strain surface identifiers.
        :type strain_surfaces: list[str]
        :param strain_surface_names: Names of strain surfaces for display.
        :type strain_surface_names: list[str]
        :param temperature_folders: Temperature folder identifiers.
        :type temperature_folders: list[str]
        :param temp_folder_names: Names of temperature folders for display.
        :type temp_folder_names: list[str]
        :param plot_both: If True, plots both LDA and GGA forces against PBE forces.
        :type plot_both: bool, optional
        :return: None
        """
    for sim_idx, sim_type in enumerate(sim_types):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        for i, strain_surface in enumerate(strain_surfaces):
            for j, temp_folder in enumerate(temperature_folders):
                data = load_npz_data(base_path, strain_surface, sim_type, temp_folder, folder='coords')

                ax = axes[i, j]
                if data is not None:
                    # Forces are stored as (50, N, 3)
                    pbe_forces_flat = data['pbe_forces'].reshape(-1) * 51.422086190832
                    lda_forces_flat = data['lda_forces'].reshape(-1) * 51.422086190832
                    gga_forces_flat = data['gga_forces'].reshape(-1) * 51.422086190832

                    if plot_both:
                        ax.scatter(pbe_forces_flat, lda_forces_flat, marker='.', alpha=0.5, label='LDA Forces')
                    ax.scatter(pbe_forces_flat, gga_forces_flat, marker='.', alpha=0.5, label='GGA Forces')

                    rmse = np.sqrt(np.mean((pbe_forces_flat - gga_forces_flat) ** 2))
                    ax.text(0.95, 0.05, f'RMSE = {rmse:.4f}', ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

                    strain_surface_name = strain_surface_names[i] if i < len(strain_surface_names) else strain_surface
                    temp_folder_name = temp_folder_names[j] if j < len(temp_folder_names) else temp_folder
                    ax.set_title(f'{strain_surface_name} - {temp_folder_name}')

                    if i == 2:
                        ax.set_xlabel('PBE Forces (eV/A)')
                    if j == 0:
                        ax.set_ylabel(f'{"LDA" if plot_both else "GGA"} Forces (eV/A)')
                else:
                    ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
                    print(strain_surface_name)
                    ax.set_title(f'{strain_surface_name} - {temp_folder_name}')

        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        file_label = "Both" if plot_both else "GGA"
        fig.savefig(os.path.join(base_path, f'Force_Comparison_{sim_type}_{file_label}.png'))
        plt.show()

def main():
    """
    The main function that generates and saves energy and force scatter plot Figures for different simulation types.

    Initializes the directory paths and parameters for strain surfaces, simulation types, and temperatures, then
    calls plotting functions to generate and save comparison plots for energies and forces across various conditions.

    :return: None
    """
    base_dir = r"../Results"
    simulation_types = ["Adsorption_Complex_Simulations", "Active_ZrH_site_simulations"]
    simulation_types_legend = ["AC_Sim", "ZrH_Sim", "Adsorption Complex Simulations", "Active ZrH Site Simulations"]
    strain_surfaces = ["min_strain_surface", "av_strain_surface", "max_strain_surface"]
    strain_surfaces_names = ["s_min", "s_av", 's_max']
    temperature_folders = ["aiMD_353K", "aiMD_773K", "Velocity_softening_dynamics"]
    temperature_folders_names = ["353K", "773K", "VSD"]

    # Create and save energy comparison plot
    plot_energies(base_path=base_dir,
                  sim_types=simulation_types,
                  sim_type_legends=simulation_types_legend,
                  strain_surfaces=strain_surfaces,
                  strain_surface_names=strain_surfaces_names,
                  temperature_folders=temperature_folders,
                  temp_folder_names=temperature_folders_names
                  )

    # Create and save force comparison plot
    plot_forces(base_path=base_dir,
                sim_types=simulation_types,
                sim_type_legends=simulation_types_legend,
                strain_surfaces=strain_surfaces,
                strain_surface_names=strain_surfaces_names,
                temperature_folders=temperature_folders,
                temp_folder_names=temperature_folders_names,
                plot_both=True
                )

if __name__ == "__main__":
    main()