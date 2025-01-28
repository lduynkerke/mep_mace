import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

def load_npz_data(base_path, strain_surface, sim_type, temp_folder, file_name):
    """
    Loads npz data for a specific strain surface, simulation type, and temperature folder.

    This function constructs the path to a npz file using the provided base path, strain surface,
    simulation type, temperature folder, and file name. It attempts to load energies and forces
    from the specified npz file. If the file is missing or an error occurs, it returns None for both.

    :param base_path: The root directory containing the data.
    :type base_path: str
    :param strain_surface: The name of the strain surface folder.
    :type strain_surface: str
    :param sim_type: The type of simulation (e.g., 'Adsorption_Complex_Simulations').
    :type sim_type: str
    :param temp_folder: The temperature folder (e.g., 'aiMD_353K').
    :type temp_folder: str
    :param file_name: The npz file name to load (e.g., 'PBE_forces.npz').
    :type file_name: str
    :return: A tuple containing energies and forces if loaded successfully, otherwise (None, None).
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    npz_path = os.path.join(base_path, strain_surface, sim_type, temp_folder, file_name)
    if os.path.exists(npz_path):
        try:
            data = np.load(npz_path)
            return data['energies'], data['forces']
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")
            return None, None
    else:
        print(f"File {npz_path} does not exist.")
        return None, None


def plot_energies(base_path, sim_types, sim_type_legends, strain_surfaces, strain_surface_names, temperature_folders,
                  temp_folder_names, file1='PBE_forces.npz', file2='mace1_forces.npz', axis_lims=None):
    """
    Generates a scatter plot comparing energies for two npz files across strain surfaces and temperature folders.

    This function generates a 3x3 grid of scatter plots, where each subplot compares the energies from
    two npz files (e.g., PBE and MACE energies) across different strain surfaces and temperature conditions.
    Colors are assigned to different simulation types. Titles are customized based on strain surface and temperature folder.

    :param base_path: The base directory containing the data.
    :type base_path: str
    :param sim_types: List of simulation types to include in the plots.
    :type sim_types: list[str]
    :param sim_type_legends: Legends corresponding to each simulation type.
    :type sim_type_legends: list[str]
    :param strain_surfaces: List of strain surfaces to include in the plots.
    :type strain_surfaces: list[str]
    :param strain_surface_names: List of display names for the strain surfaces.
    :type strain_surface_names: list[str]
    :param temperature_folders: List of temperature folders to include in the plots.
    :type temperature_folders: list[str]
    :param temp_folder_names: List of display names for the temperature folders.
    :type temp_folder_names: list[str]
    :param file1: First npz file to compare (default is 'PBE_forces.npz').
    :type file1: str
    :param file2: Second npz file to compare (default is 'mace0_forces.npz').
    :type file2: str
    :param axis_lims: Axis limits to use for the scatter plot.
    :type axis_lims: tuple(tuple(float, float), tuple(float, float), list[float], list[float])
    """
    for sim_idx, sim_type in enumerate(sim_types):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        # fig.suptitle("Energy Comparison")
        for i, strain_surface in enumerate(strain_surfaces):
            for j, temp_folder in enumerate(temperature_folders):
                pbe_energies, pbe_forces = load_npz_data(base_path, strain_surface, sim_type, temp_folder, file1)
                mace_energies, _ = load_npz_data(base_path, strain_surface, sim_type, temp_folder, file2)

                ax = axes[i, j]

                if pbe_energies is not None and mace_energies is not None:
                    _, _, atoms = pbe_forces.shape
                    x, y = pbe_energies/atoms*27.2114079527, mace_energies/atoms*27.2114079527
                    ax.scatter(x, y, label=f'{sim_type_legends[sim_idx]} Energies')

                    # Add RMSE
                    rmse = np.sqrt(np.mean((x - y) ** 2))
                    ax.text(0.95, 0.10, f'RMSE = {rmse:.4f}',
                            ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

                    # Add trendline and formula
                    slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
                    sns.regplot(x=x, y=y, ax=ax, scatter=False, color='r')
                    ax.text(0.95, 0.05, 'y = ' + str(round(intercept, 3)) + ' + ' + str(round(slope, 3)) + 'x',
                             ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

                    # Set custom titles using strain_surface_names and temp_folder_names
                    strain_surface_name = strain_surface_names[i] if i < len(strain_surface_names) else strain_surface
                    temp_folder_name = temp_folder_names[j] if j < len(temp_folder_names) else temp_folder
                    ax.set_title(f'{strain_surface_name} - {temp_folder_name}')

                    # Only add labels on the outer axes
                    if i == 2:
                        ax.set_xlabel('PBE Energy / atom (eV)')
                    if j == 0:
                        ax.set_ylabel('MACE-MP-0 Energy / atom (eV)')

                    if axis_lims is not None:
                        ax.set_xlim(axis_lims[sim_idx][0])
                        ax.set_ylim(axis_lims[sim_idx][1])
                        ax.set_xticks(axis_lims[sim_idx][2])
                        ax.set_yticks(axis_lims[sim_idx][3])

                else:
                    ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
                    ax.set_title(f'{strain_surface_name} - {temp_folder_name}')

                # ax.grid(False)

        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))

        fig.savefig(os.path.join(base_path, f'Energy_Comparison_{sim_type}_{file2.split('_')[0]}.png'))
        plt.show()

def plot_forces(base_path, sim_types, sim_type_legends, strain_surfaces, strain_surface_names, temperature_folders,
                temp_folder_names, file1='PBE_forces.npz', file2='mace1_forces.npz'):
    """
    Generates scatter plots comparing forces for npz files across strain surfaces and temperature folders.

    This function generates a 3x3 grid of scatter plots for each simulation type, where each subplot
    compares the forces from two npz files (e.g., PBE and MACE forces). The forces are flattened, and
    a single color is used for all x, y, z components. Titles and axes are customized based on strain
    surfaces and temperature folders.

    :param base_path: The base directory containing the data.
    :type base_path: str
    :param sim_types: List of simulation types to include in the plots.
    :type sim_types: list[str]
    :param sim_type_legends: Legends corresponding to each simulation type.
    :type sim_type_legends: list[str]
    :param strain_surfaces: List of strain surfaces to include in the plots.
    :type strain_surfaces: list[str]
    :param strain_surface_names: List of display names for the strain surfaces.
    :type strain_surface_names: list[str]
    :param temperature_folders: List of temperature folders to include in the plots.
    :type temperature_folders: list[str]
    :param temp_folder_names: List of display names for the temperature folders.
    :type temp_folder_names: list[str]
    :param file1: First npz file to compare (default is 'PBE_forces.npz').
    :type file1: str
    :param file2: Third npz file to compare (default is 'mace1_forces.npz').
    :type file2: str
    """
    for sim_idx, sim_type in enumerate(sim_types):
        fig, axes = plt.subplots(3, 3, figsize=(15, 15))
        # fig.suptitle(f"Force Comparison for {sim_type_legends[sim_idx+2]}")

        for i, strain_surface in enumerate(strain_surfaces):
            for j, temp_folder in enumerate(temperature_folders):
                _, pbe_forces = load_npz_data(base_path, strain_surface, sim_type, temp_folder, file1)
                _, mace_forces = load_npz_data(base_path, strain_surface, sim_type, temp_folder, file2)
                pbe_forces = np.swapaxes(pbe_forces, 1, 2)

                ax = axes[i, j]

                if pbe_forces is not None and mace_forces is not None:
                    pbe_forces_flat = pbe_forces.reshape(-1)*51.422086190832     # Flatten and convert Eh/B -> eV/A
                    mace_forces_flat = mace_forces.reshape(-1)*51.422086190832   # Flatten and convert Eh/B -> eV/A
                    slope, intercept, _, _, _ = scipy.stats.linregress(pbe_forces_flat, mace_forces_flat)
                    rmse = np.sqrt(np.mean((pbe_forces_flat - mace_forces_flat) ** 2))
                    _, _, natoms = pbe_forces.shape

                    ax.scatter(pbe_forces_flat, mace_forces_flat, marker='.', alpha=0.5, label=f'mace1')

                    # Add RMSE
                    rmse = np.sqrt(np.mean((pbe_forces_flat - mace_forces_flat) ** 2))
                    ax.text(0.95, 0.10, f'RMSE = {rmse:.4f}',
                            ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

                    # Add trendline and formula
                    slope, intercept, _, _, _ = scipy.stats.linregress(pbe_forces_flat, mace_forces_flat)
                    sns.regplot(x=pbe_forces_flat, y=mace_forces_flat, ax=ax, scatter=False, color='r')
                    ax.text(0.95, 0.05, 'y = ' + str(round(intercept, 3)) + ' + ' + str(round(slope, 3)) + 'x',
                             ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

                    # Set custom titles using strain_surface_names and temp_folder_names
                    strain_surface_name = strain_surface_names[i] if i < len(strain_surface_names) else strain_surface
                    temp_folder_name = temp_folder_names[j] if j < len(temp_folder_names) else temp_folder
                    ax.set_title(f'{strain_surface_name} - {temp_folder_name}')
                    ax.set_ylim(-10, 10)
                    ax.set_xlim(-10, 10)

                    # Only add labels on the outer axes
                    if i == 2:
                        ax.set_xlabel('PBE Forces (eV/A)')
                    if j == 0:
                        ax.set_ylabel('MACE-MP-0 Forces (eV/A)')

                else:
                    ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
                    ax.set_title(f'{strain_surface_name} - {temp_folder_name}')

                ax.grid(False)
                print(i,j)

        plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))
        fig.savefig(os.path.join(base_path, f'Force_Comparison_{sim_type_legends[sim_idx]}.png'))

    plt.show()

def plot_slides():
    dir = '../../Results/av_strain_surface/Adsorption_Complex_Simulations/Velocity_softening_dynamics'
    labels = ['MACE-MP-0', 'MACE1']
    pbe_energies, pbe_forces = load_npz_data(
        base_path='../../Results',
        strain_surface='av_strain_surface',
        sim_type='Adsorption_Complex_Simulations',
        temp_folder='Velocity_softening_dynamics',
        file_name='PBE_forces.npz'
    )
    mace0_energies, mace0_forces = load_npz_data(
        base_path='../../Results',
        strain_surface='av_strain_surface',
        sim_type='Adsorption_Complex_Simulations',
        temp_folder='Velocity_softening_dynamics',
        file_name='mace0_forces.npz'
    )
    # mace1_energies, mace1_forces = load_npz_data(
    #     base_path='../../Results',
    #     strain_surface='av_strain_surface',
    #     sim_type='Adsorption_Complex_Simulations',
    #     temp_folder='Velocity_softening_dynamics',
    #     file_name='mace_large_forces.npz'
    # )
    mace1_energies, mace1_forces = load_npz_data(
        base_path='../../Results',
        strain_surface='max_strain_surface',
        sim_type='Active_ZrH_site_simulations',
        temp_folder='aiMD_773K',
        file_name='mace1_forces.npz'
    )
    pbe_forces = np.swapaxes(pbe_forces, 1, 2)
    _, atoms, _ = pbe_forces.shape

    # Energies
    for i, data in enumerate([mace0_energies, mace1_energies]):
        x, y = pbe_energies / atoms * 27.2114079527, data / atoms * 27.2114079527
        slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
        rmse = np.sqrt(np.mean((x - y) ** 2))

        #plt.plot(x, y), mace_energies)
        plt.scatter(x, y)#, label=f'MACE 1 Energies')
        sns.regplot(x=x, y=y, scatter=False, color='r')
        plt.text(0.95, 0.10, f'RMSE = {rmse:.4f}',
                ha='right', va='bottom', fontsize=12, transform=plt.gca().transAxes)
        plt.text(0.95, 0.05, 'y = ' + str(round(intercept, 3)) + ' + ' + str(round(slope, 3)) + 'x',
                ha='right', va='bottom', fontsize=12, transform=plt.gca().transAxes)
        plt.xlabel('PBE Energy / atom (eV)')
        plt.ylabel(labels[i]+' Energy / atom (eV)')
        plt.tight_layout()
        plt.show()

    # Forces
    pbe_forces_flat = pbe_forces.reshape(-1)*51.422086190832      # Flatten and convert Eh/B -> eV/A
    mace0_forces_flat = mace0_forces.reshape(-1)*51.422086190832  # Flatten and convert Eh/B -> eV/A
    mace1_forces_flat = mace1_forces.reshape(-1)*51.422086190832  # Flatten and convert Eh/B -> eV/A
    slope0, intercept0, _, _, _ = scipy.stats.linregress(pbe_forces_flat, mace0_forces_flat)
    slope1, intercept1, _, _, _ = scipy.stats.linregress(pbe_forces_flat, mace1_forces_flat)
    rmse0 = np.sqrt(np.mean((pbe_forces_flat - mace0_forces_flat) ** 2))
    rmse1 = np.sqrt(np.mean((pbe_forces_flat - mace1_forces_flat) ** 2))
    # rmse2 = np.sqrt(np.mean((norm_forces_mace - norm_forces_pbe) ** 2))

    plt.figure(figsize=(5,5))
    plt.scatter(pbe_forces_flat, mace0_forces_flat, marker='.', alpha=0.5, label=f'mace0')
    sns.regplot(x=pbe_forces_flat, y=mace0_forces_flat, scatter=False, color='r')
    plt.text(0.95, 0.10, f'RMSE = {rmse0:.4f}',
             ha='right', va='bottom', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.95, 0.05, 'y = ' + str(round(intercept0, 3)) + ' + ' + str(round(slope0, 3)) + 'x',
             ha='right', va='bottom', fontsize=12, transform=plt.gca().transAxes)
    plt.ylim(-600, 600)
    plt.xlim(-15, 15)
    plt.xlabel('PBE Forces (eV/Å)')
    plt.ylabel('MACE Forces (eV/Å)')
    plt.tight_layout()
    #plt.legend()
    plt.show()

    plt.figure(figsize=(5,5))
    plt.scatter(pbe_forces_flat, mace1_forces_flat, marker='.', alpha=0.5, label=f'mace1')
    sns.regplot(x=pbe_forces_flat, y=mace1_forces_flat, scatter=False, color='r')
    plt.text(0.95, 0.10, f'RMSE = {rmse1:.4f}',
             ha='right', va='bottom', fontsize=12, transform=plt.gca().transAxes)
    plt.text(0.95, 0.05, 'y = ' + str(round(intercept1, 3)) + ' + ' + str(round(slope1, 3)) + 'x',
             ha='right', va='bottom', fontsize=12, transform=plt.gca().transAxes)
    plt.ylim(-15, 15)
    plt.xlim(-15, 15)
    plt.xlabel('PBE Forces (eV/Å)')
    plt.ylabel('MACE Forces (eV/Å)')
    plt.tight_layout()
    plt.show()


def main():
    """
    The main function that generates and saves energy and force scatter plot figures for different simulation types.
    """
    base_dir = r"../../Results"
    simulation_types = ["Adsorption_Complex_Simulations", "Active_ZrH_site_simulations"]
    simulation_types_legend = ["AC_Sim", "ZrH_Sim", "Adsorption Complex Simulations", "Active ZrH Site Simulations"]
    strain_surfaces = ["min_strain_surface", "av_strain_surface", "max_strain_surface"]
    strain_surfaces_names = ["s_min", "s_av", 's_max']
    temperature_folders = ["aiMD_353K", "aiMD_773K", "Velocity_softening_dynamics"]
    temperature_folders_names = ["353K", "773K", "VSD"]

    #plot_slides()
    #
    # # Create and save energy comparison plot
    # plot_energies(base_path=base_dir,
    #               sim_types=simulation_types,
    #               sim_type_legends=simulation_types_legend,
    #               strain_surfaces=strain_surfaces,
    #               strain_surface_names=strain_surfaces_names,
    #               temperature_folders=temperature_folders,
    #               temp_folder_names=temperature_folders_names
    #               )
    #
    plot_energies(
        base_path=base_dir,
        sim_types=simulation_types,
        sim_type_legends=simulation_types_legend,
        strain_surfaces=strain_surfaces,
        strain_surface_names=strain_surfaces_names,
        temperature_folders=temperature_folders,
        temp_folder_names=temperature_folders_names,
        file2='mace3_forces.npz'
        )
    #
    # # Create and save force comparison plot
    # plot_forces(base_path=base_dir,
    #             sim_types=simulation_types,
    #             sim_type_legends=simulation_types_legend,
    #             strain_surfaces=strain_surfaces,
    #             strain_surface_names=strain_surfaces_names,
    #             temperature_folders=temperature_folders,
    #             temp_folder_names=temperature_folders_names
    #             )

    plot_forces(
        base_path=base_dir,
        sim_types=simulation_types,
        sim_type_legends=["AC_Sim", "ZrH_Sim", "Adsorption Complex Simulations", "Active ZrH Site Simulations"],
        strain_surfaces=strain_surfaces,
        strain_surface_names=strain_surfaces_names,
        temperature_folders=temperature_folders,
        temp_folder_names=temperature_folders_names,
        file2='mace3_forces.npz'
        )

if __name__ == "__main__":
    main()