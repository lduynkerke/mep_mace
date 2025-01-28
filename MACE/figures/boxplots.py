import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def reshape_forces(pbe_forces):
    """
    Reshape forces to ensure they are comparable between PBE and MACE force arrays.
    Transposes PBE forces if their shape is (603, 3, 506), (606, 3, 506), or (603, 3, 504)
    to match the expected shape (603, 506, 3).

    :param pbe_forces: The PBE forces array to reshape.
    :type pbe_forces: np.ndarray
    :return: Reshaped PBE forces array.
    :rtype: np.ndarray
    """
    if pbe_forces.shape == (603, 3, 506) or pbe_forces.shape == (606, 3, 506) or pbe_forces.shape == (603, 3, 504):
        pbe_forces = np.transpose(pbe_forces, (0, 2, 1))  # Transpose PBE forces to match shape (603, 506, 3)
    return pbe_forces

def load_npz_data(base_path, strain_surface, sim_type, temp_folder, file):
    """
      Load energies and forces from a given .npz file.

      :param base_path: The base directory containing the data.
      :type base_path: str
      :param strain_surface: The strain surface subdirectory.
      :type strain_surface: str
      :param sim_type: The type of simulation (e.g., ZrH or Adsorption Complex).
      :type sim_type: str
      :param temp_folder: The temperature folder subdirectory.
      :type temp_folder: str
      :param file: The .npz file name to load.
      :type file: str
      :return: Tuple containing the loaded energies and forces, or (None, None) if the file does not exist.
      :rtype: tuple
      """
    npz_path = os.path.join(base_path, strain_surface, sim_type, temp_folder, file)
    if os.path.exists(npz_path):
        data = np.load(npz_path)
        return data['energies'], data['forces']
    return None, None

def compute_errors(pbe_data, mace_data):
    """
    Compute the absolute errors between PBE and MACE data for both energies and forces.

    :param pbe_data: A tuple containing PBE energies and PBE forces.
    :type pbe_data: tuple
    :param mace_data: A tuple containing MACE energies and MACE forces.
    :type mace_data: tuple
    :return: A tuple containing the energy and force errors.
    :rtype: tuple
    """
    energy_diff = np.abs(mace_data[0] - pbe_data[0]) if mace_data[0] is not None and pbe_data[0] is not None else None
    force_diff = np.abs(mace_data[1].reshape(-1) - pbe_data[1].reshape(-1)) if mace_data[1] is not None and pbe_data[1] is not None else None

    return energy_diff, force_diff

def plot_errors_by_simulation(base_path, strain_surfaces, sim_types, temp_folders,
                              strain_surface_names, temp_folder_names, file1):
    """
    Plot errors for energies and forces in separate subplots for ZrH and Adsorption Complex simulations.
    Each simulation type has two subplots (mace0 and mace1), with one figure for energies and another for forces.
    The x-axis contains the combinations of strain surfaces and temperature folders.

    :param base_path: The base directory containing the result data.
    :type base_path: str
    :param strain_surfaces: A list of strain surface subdirectories.
    :type strain_surfaces: list
    :param sim_types: A list of simulation types (e.g., ZrH or AC simulations).
    :type sim_types: list
    :param temp_folders: A list of temperature folder subdirectories.
    :type temp_folders: list
    :param strain_surface_names: A list of names for the strain surfaces for labeling purposes.
    :type strain_surface_names: list
    :param temp_folder_names: A list of names for the temperature folders for labeling purposes.
    :type temp_folder_names: list
    :param file1: The file name of the reference .npz file (PBE forces).
    :type file1: str
    """
    for sim_type, sim_label, titles in zip(sim_types, ['ZrH_Sim', 'AC_Sim'],
                                           ['Active ZrH Site Simulations', 'Adsorption Complex Simulations']):
        fig_energy, axes_energy = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        fig_forces, axes_forces = plt.subplots(2, 1, figsize=(12, 12), sharex=True)
        fig_energy.suptitle(f'{titles} Energy Errors')
        fig_forces.suptitle(f'{titles} Force Errors')

        for ax_idx, (file, title) in enumerate(zip(['mace0_forces.npz', 'mace1_24_forces.npz'],
                                                   ['Mace-MP-0 Errors', 'Mace1 Errors'])):
            energy_errors, force_errors = [], []

            for strain_surface, strain_surface_name in zip(strain_surfaces, strain_surface_names):
                for temp_folder, temp_folder_name in zip(temp_folders, temp_folder_names):
                    pbe_energies_raw, pbe_forces_raw = load_npz_data(base_path, strain_surface, sim_type, temp_folder,
                                                             file1)
                    mace_energies_raw, mace_forces_raw = load_npz_data(base_path, strain_surface, sim_type, temp_folder,
                                                               file)

                    atoms = np.shape(mace_forces_raw)[1]
                    pbe_forces = reshape_forces(pbe_forces_raw)*51.422086190832 # from Eh/B to eV/A
                    pbe_energies = pbe_energies_raw/atoms*27.2114079527         # from Eh to eV
                    mace_forces = mace_forces_raw*51.422086190832               # from Eh/B to eV/A
                    mace_energies = mace_energies_raw/atoms*27.2114079527       # from Eh to eV

                    if pbe_energies is not None and mace_energies is not None:
                        min_threshold = 1e-5
                        energy_err, force_err = compute_errors((pbe_energies, pbe_forces),
                                                               (mace_energies, mace_forces))
                        #filtered_energy_err = err for err in energy_err if abs(err) >= min_threshold])
                        #filtered_force_err = np.log1p([err for err in force_err if abs(err) >= min_threshold])
                        filtered_force_err = np.log1p(force_err)
                        energy_errors.append(energy_err)
                        force_errors.append(filtered_force_err)

            axes_energy[ax_idx].boxplot(energy_errors, patch_artist=True)
            axes_energy[ax_idx].set_title(title)
            #sns.boxenplot(force_errors, ax=axes_forces[ax_idx])
            axes_forces[ax_idx].boxplot(force_errors, patch_artist=True)
            axes_forces[ax_idx].set_title(title)
            if ax_idx == 0:
                axes_energy[ax_idx].set_ylim(94, 100)

        for ax in axes_energy:
            ax.set_xticks(range(1, 10))
            ax.set_xticklabels([f'{s}_{t}' for s in strain_surface_names for t in temp_folder_names], rotation=45)
            ax.set_ylabel('Energy Errors / Atom (eV)')

        for ax in axes_forces:
            print(ax)
            ax.set_xticks(range(1, 10))
            ax.set_xticklabels([f'{s}_{t}' for s in strain_surface_names for t in temp_folder_names], rotation=45)
            ax.set_ylabel('Force Errors (eV/A)')
            #ax.set_yscale('log')
            #ax.set_ylim(1e-6, 1e4)

        fig_energy.savefig(os.path.join(base_path, f'{sim_label}_energy_errors.png'))
        fig_forces.savefig(os.path.join(base_path, f'{sim_label}_force_errors.png'))
        plt.show()


def plot_errors(base_path, strain_surfaces, sim_types, temp_folders, strain_surface_names, temp_folder_names, file1):
    """
    Plot errors for energies and forces in separate subplots for ZrH and Adsorption Complex simulations.
    Each simulation type has two subplots (mace0 and mace1), with one figure for energies and another for forces.
    The x-axis contains the combinations of strain surfaces and temperature folders.

    :param base_path: The base directory containing the result data.
    :type base_path: str
    :param strain_surfaces: A list of strain surface subdirectories.
    :type strain_surfaces: list
    :param sim_types: A list of simulation types (e.g., ZrH or AC simulations).
    :type sim_types: list
    :param temp_folders: A list of temperature folder subdirectories.
    :type temp_folders: list
    :param strain_surface_names: A list of names for the strain surfaces for labeling purposes.
    :type strain_surface_names: list
    :param temp_folder_names: A list of names for the temperature folders for labeling purposes.
    :type temp_folder_names: list
    :param file1: The file name of the reference .npz file (PBE forces).
    :type file1: str
    """
    for sim_type, sim_label, titles in zip(sim_types, ['ZrH_Sim', 'AC_Sim'],
                                           ['CA Simulations', 'PA Simulations']):
        energy_errors, force_errors = [], []

        for strain_surface, strain_surface_name in zip(strain_surfaces, strain_surface_names):
            for temp_folder, temp_folder_name in zip(temp_folders, temp_folder_names):
                pbe_energies_raw, pbe_forces_raw = load_npz_data(base_path, strain_surface, sim_type, temp_folder,
                                                         file1)
                mace_energies_raw, mace_forces_raw = load_npz_data(base_path, strain_surface, sim_type, temp_folder,
                                                           'mace3_forces.npz')
                atoms = np.shape(mace_forces_raw)[1]
                pbe_forces = reshape_forces(pbe_forces_raw)*51.422086190832 # from Eh/B to eV/A
                pbe_energies = pbe_energies_raw/atoms*27.2114079527         # from Eh to eV
                mace_forces = mace_forces_raw*51.422086190832               # from Eh/B to eV/A
                mace_energies = mace_energies_raw/atoms*27.2114079527       # from Eh to eV

                if pbe_energies is not None and mace_energies is not None:
                    energy_err, force_err = compute_errors((pbe_energies, pbe_forces),
                                                           (mace_energies, mace_forces))
                    # filtered_force_err = [err for err in force_err if abs(err) >= 1e-5]
                    filtered_force_err = np.log10([err for err in force_err if abs(err) >= 1e-3])
                    energy_errors.append(energy_err)
                    force_errors.append(filtered_force_err)

        plt.figure()
        plt.boxplot(energy_errors, patch_artist=True)
        #plt.title(titles+' Errors')
        plt.xticks(range(1, 10), labels=[f'{s}_{t}' for s in strain_surface_names for t in temp_folder_names], rotation=45)
        plt.ylabel('Energy Errors (eV)')
        plt.tight_layout()
        plt.savefig(sim_label+'_energy_err.png')
        plt.show()

        # plt.figure()
        # sns.violinplot(force_errors)
        # #plt.boxplot(force_errors, patch_artist=True)
        # #plt.title(titles+' Errors')
        # plt.xticks(range(0, 9), labels=[f'{s}_{t}' for s in strain_surface_names for t in temp_folder_names], rotation=45)
        # plt.yticks([-3, -2, -1, 0, 1])
        # plt.ylabel('10 log Force Errors (eV/Å)')
        # plt.tight_layout()
        # plt.savefig(sim_label+'_force_err.png')
        # plt.show()

            # plt.boxplot(force_errors, patch_artist=True)
            # plt.set_title('Mace1 Errors')

        # for ax in axes_energy:
        #     ax.set_xticks(range(1, 10))
        #     ax.set_xticklabels([f'{s}_{t}' for s in strain_surface_names for t in temp_folder_names], rotation=45)
        #     ax.set_ylabel('Energy Errors / Atom (eV)')
        #
        # for ax in axes_forces:
        #     print(ax)
        #     ax.set_xticks(range(1, 10))
        #     ax.set_xticklabels([f'{s}_{t}' for s in strain_surface_names for t in temp_folder_names], rotation=45)
        #     ax.set_ylabel('Force Errors (eV/A)')
        #     ax.set_yscale('log')
        #     ax.set_ylim(1e-6, 1e4)
        #
        # fig_energy.savefig(os.path.join(base_path, f'{sim_label}_energy_errors.png'))
        # fig_forces.savefig(os.path.join(base_path, f'{sim_label}_force_errors.png'))
        plt.show()


def main():
    """
    Main function to define parameters for plotting energy and force errors and call the plotting function.
    """
    base_dir = "../../Results"
    strain_surfaces = ["min_strain_surface", "av_strain_surface", "max_strain_surface"]
    strain_surface_names = ["s_min", "s_av", "s_max"]
    temperature_folders = ["aiMD_353K", "aiMD_773K", "Velocity_softening_dynamics"]
    temperature_folders_names = ["353K", "773K", "VSD"]
    sim_types = ["Active_ZrH_site_simulations", "Adsorption_Complex_Simulations"]
    file1 = "PBE_forces.npz"

    # plot_errors_by_simulation(base_dir, strain_surfaces, sim_types, temperature_folders,
    #                           strain_surface_names, temperature_folders_names, file1)

    plot_errors(base_dir, strain_surfaces, sim_types, temperature_folders,
                               strain_surface_names, temperature_folders_names, file1)


if __name__ == '__main__':
    main()
