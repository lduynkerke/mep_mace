import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy

def load_npz_data(full_path):
    """
    Loads npz data from a given path.

    This function attempts to load energies and forces from the specified npz file.
    If the file is missing or an error occurs, it returns None for both.

    :param full_path: Full path to the npz file.
    :type full_path: str
    :return: A tuple containing energies and forces if loaded successfully, otherwise (None, None).
    :rtype: tuple[np.ndarray, np.ndarray]
    """
    if os.path.exists(full_path):
        try:
            data = np.load(full_path)
            return data['energies'], data['forces']
        except Exception as e:
            print(f"Error loading {full_path}: {e}")
            return None, None
    else:
        print(f"File {full_path} does not exist.")
        return None, None


def plot_energies(data_paths, labels, file1='PBE_forces.npz', file2='mace1_forces.npz', axis_lims=None):
    """
    Generates scatter plots comparing energies for two npz files across provided data paths.

    :param data_paths: List of full paths to the data directories.
    :type data_paths: list[str]
    :param labels: List of labels for each dataset (for plot titles and legends).
    :type labels: list[str]
    :param file1: First npz file to compare (default is 'PBE_forces.npz').
    :type file1: str
    :param file2: Second npz file to compare (default is 'mace1_forces.npz').
    :type file2: str
    :param axis_lims: Axis limits to use for the scatter plot.
    :type axis_lims: tuple(tuple(float, float), tuple(float, float), list[float], list[float])
    """
    for i, (path, label) in enumerate(zip(data_paths, labels)):
        fig, ax = plt.subplots(figsize=(8, 8))
        pbe_energies, pbe_forces = load_npz_data(os.path.join(path, file1))
        mace_energies, _ = load_npz_data(os.path.join(path, file2))

        if pbe_energies is not None and mace_energies is not None:
            _, atoms, _ = pbe_forces.shape
            x, y = pbe_energies / atoms * 27.2114079527, mace_energies / atoms * 27.2114079527
            ax.scatter(x, y, label=f'{label} Energies')

            # Add RMSE
            rmse = np.sqrt(np.mean((x - y) ** 2))
            ax.text(0.95, 0.10, f'RMSE = {rmse:.4f}', ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

            # Add trendline and formula
            slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
            sns.regplot(x=x, y=y, ax=ax, scatter=False, color='r')
            ax.text(0.95, 0.05, f'y = {round(intercept, 3)} + {round(slope, 3)}x',
                     ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

            ax.set_title(f'Energy Comparison for {label}')
            ax.set_xlabel('PBE Energy / atom (eV)')
            ax.set_ylabel('MACE Energy / atom (eV)')

            if axis_lims is not None:
                ax.set_xlim(axis_lims[0])
                ax.set_ylim(axis_lims[1])
                ax.set_xticks(axis_lims[2])
                ax.set_yticks(axis_lims[3])
        else:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
            ax.set_title(f'{label} - No Data')

        plt.tight_layout()
        plt.savefig(os.path.join(path, f'Energy_Comparison_{label}_{file2.split("_")[0]}.png'))
        plt.show()


def plot_forces(data_paths, labels, file1='PBE_forces.npz', file2='mace1_forces.npz'):
    """
    Generates scatter plots comparing forces for npz files across provided data paths.

    :param data_paths: List of full paths to the data directories.
    :type data_paths: list[str]
    :param labels: List of labels for each dataset (for plot titles and legends).
    :type labels: list[str]
    :param file1: First npz file to compare (default is 'PBE_forces.npz').
    :type file1: str
    :param file2: Second npz file to compare (default is 'mace1_forces.npz').
    :type file2: str
    """
    for i, (path, label) in enumerate(zip(data_paths, labels)):
        fig, ax = plt.subplots(figsize=(8, 8))
        _, pbe_forces = load_npz_data(os.path.join(path, file1))

        _, mace_forces = load_npz_data(os.path.join(path, file2))
        #pbe_forces = np.swapaxes(pbe_forces, 1, 2)#[:189, :, :]

        if pbe_forces is not None and mace_forces is not None:
            pbe_forces_flat = pbe_forces.reshape(-1) * 51.422086190832  # Convert to eV/Å
            mace_forces_flat = mace_forces.reshape(-1) * 51.422086190832  # Convert to eV/Å
            slope, intercept, _, _, _ = scipy.stats.linregress(pbe_forces_flat, mace_forces_flat)
            rmse = np.sqrt(np.mean((pbe_forces_flat - mace_forces_flat) ** 2))

            ax.scatter(pbe_forces_flat, mace_forces_flat, marker='.', alpha=0.5, label=f'{label} Forces')
            sns.regplot(x=pbe_forces_flat, y=mace_forces_flat, ax=ax, scatter=False, color='r')
            ax.text(0.95, 0.10, f'RMSE = {rmse:.4f}', ha='right', va='bottom', transform=ax.transAxes, fontsize=12)
            ax.text(0.95, 0.05, f'y = {round(intercept, 3)} + {round(slope, 3)}x',
                     ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

            ax.set_title(f'Force Comparison for {label}')
            ax.set_xlabel('PBE Forces (eV/Å)')
            ax.set_ylabel('MACE Forces (eV/Å)')
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
        else:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
            ax.set_title(f'{label} - No Data')

        plt.tight_layout()
        plt.savefig(os.path.join(path, f'Force_Comparison_{label}.png'))
        plt.show()


def plot_energies(data_paths, labels, file1='PBE_forces.npz', file2='mace1_forces.npz', axis_lims=None,
                  subplot_shape=None):
    """
    Generates scatter plots comparing energies for two npz files across provided data paths. Optionally generates subplots.

    :param data_paths: List of full paths to the data directories.
    :type data_paths: list[str]
    :param labels: List of labels for each dataset (for plot titles and legends).
    :type labels: list[str]
    :param file1: First npz file to compare (default is 'PBE_forces.npz').
    :type file1: str
    :param file2: Second npz file to compare (default is 'mace1_forces.npz').
    :type file2: str
    :param axis_lims: Axis limits to use for the scatter plot.
    :type axis_lims: tuple(tuple(float, float), tuple(float, float), list[float], list[float])
    :param subplot_shape: Shape of subplots grid (rows, cols). If None, generates individual plots.
    :type subplot_shape: tuple[int, int] or None
    """
    if subplot_shape:
        fig, axes = plt.subplots(*subplot_shape, figsize=(8 * subplot_shape[1], 8 * subplot_shape[0]))
        axes = axes.flatten()

    for i, (path, label) in enumerate(zip(data_paths, labels)):
        ax = axes[i] if subplot_shape else plt.subplots(figsize=(8, 8))[1]

        pbe_energies, pbe_forces = load_npz_data(os.path.join(path, file1))
        mace_energies, _ = load_npz_data(os.path.join(path, file2))

        if pbe_energies is not None and mace_energies is not None:
            _, atoms, _ = pbe_forces.shape
            x, y = pbe_energies / atoms * 27.2114079527, mace_energies / atoms * 27.2114079527
            ax.scatter(x, y, label=f'{label} Energies')

            rmse = np.sqrt(np.mean((x - y) ** 2))
            ax.text(0.95, 0.10, f'RMSE = {rmse:.4f}', ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

            slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
            sns.regplot(x=x, y=y, ax=ax, scatter=False, color='r')
            ax.text(0.95, 0.05, f'y = {round(intercept, 3)} + {round(slope, 3)}x',
                    ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

            ax.set_title(f'Energy Comparison for {label}')
            ax.set_xlabel('PBE Energy / atom (eV)')
            ax.set_ylabel('MACE Energy / atom (eV)')

            if axis_lims is not None:
                ax.set_xlim(axis_lims[0])
                ax.set_ylim(axis_lims[1])
                if len(axis_lims) > 2:
                    ax.set_xticks(axis_lims[2])
                    ax.set_yticks(axis_lims[3])
        else:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
            ax.set_title(f'{label} - No Data')

        if not subplot_shape:
            plt.tight_layout()
            plt.savefig(os.path.join(path, f'Energy_Comparison_{label}_{file2.split("_")[0]}.png'))
            plt.show()

    if subplot_shape:
        plt.tight_layout()
        plt.savefig('Energy_Comparison_Subplots.png')
        plt.show()


def plot_forces(data_paths, labels, file1='PBE_forces.npz', file2='mace1_forces.npz', subplot_shape=None):
    """
    Generates scatter plots comparing forces for npz files across provided data paths. Optionally generates subplots.

    :param data_paths: List of full paths to the data directories.
    :type data_paths: list[str]
    :param labels: List of labels for each dataset (for plot titles and legends).
    :type labels: list[str]
    :param file1: First npz file to compare (default is 'PBE_forces.npz').
    :type file1: str
    :param file2: Second npz file to compare (default is 'mace1_forces.npz').
    :type file2: str
    :param subplot_shape: Shape of subplots grid (rows, cols). If None, generates individual plots.
    :type subplot_shape: tuple[int, int] or None
    """
    if subplot_shape:
        fig, axes = plt.subplots(*subplot_shape, figsize=(8 * subplot_shape[1], 8 * subplot_shape[0]))
        axes = axes.flatten()

    for i, (path, label) in enumerate(zip(data_paths, labels)):
        ax = axes[i] if subplot_shape else plt.subplots(figsize=(8, 8))[1]

        _, pbe_forces = load_npz_data(os.path.join(path, file1))
        _, mace_forces = load_npz_data(os.path.join(path, file2))

        if pbe_forces is not None and mace_forces is not None:
            pbe_forces_flat = pbe_forces.reshape(-1) * 51.422086190832  # Convert to eV/\u00c5
            mace_forces_flat = mace_forces.reshape(-1) * 51.422086190832  # Convert to eV/\u00c5
            slope, intercept, _, _, _ = scipy.stats.linregress(pbe_forces_flat, mace_forces_flat)
            rmse = np.sqrt(np.mean((pbe_forces_flat - mace_forces_flat) ** 2))

            ax.scatter(pbe_forces_flat, mace_forces_flat, marker='.', alpha=0.5, label=f'{label} Forces')
            sns.regplot(x=pbe_forces_flat, y=mace_forces_flat, ax=ax, scatter=False, color='r')
            ax.text(0.95, 0.10, f'RMSE = {rmse:.4f}', ha='right', va='bottom', transform=ax.transAxes, fontsize=12)
            ax.text(0.95, 0.05, f'y = {round(intercept, 3)} + {round(slope, 3)}x',
                    ha='right', va='bottom', transform=ax.transAxes, fontsize=12)

            ax.set_title(f'Force Comparison for {label}')
            ax.set_xlabel('PBE Forces (eV/\u00c5)')
            ax.set_ylabel('MACE Forces (eV/\u00c5)')
            ax.set_xlim(-15, 15)
            ax.set_ylim(-15, 15)
        else:
            ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
            ax.set_title(f'{label} - No Data')

        if not subplot_shape:
            plt.tight_layout()
            plt.savefig(os.path.join(path, f'Force_Comparison_{label}.png'))
            plt.show()

    if subplot_shape:
        plt.tight_layout()
        plt.savefig('Force_Comparison_Subplots.png')
        plt.show()

def plot_energy_force_combined(data_path, label, file1='PBE_forces.npz', file2='mace1_forces.npz',
                               axis_lims_e=None, axis_lims_f=None):
    """
    Generates a combined plot of energy and force comparisons for a single dataset.

    :param data_path: Path to the data directory.
    :type data_path: str
    :param label: Label for the dataset (for plot titles and legends).
    :type label: str
    :param file1: First npz file to compare (default is 'PBE_forces.npz').
    :type file1: str
    :param file2: Second npz file to compare (default is 'mace1_forces.npz').
    :type file2: str
    :param axis_lims_e: Axis limits to use for the energy scatter plot.
    :type axis_lims_e: tuple(tuple(float, float), tuple(float, float), list[float], list[float])
    :param axis_lims_f: Axis limits to use for the force scatter plot.
    :type axis_lims_f: tuple(tuple(float, float), tuple(float, float), list[float], list[float])
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Energy plot
    pbe_energies, pbe_forces = load_npz_data(os.path.join(data_path, file1))
    mace_energies, mace_forces = load_npz_data(os.path.join(data_path, file2))

    if pbe_energies is not None and mace_energies is not None:
        _, atoms, _ = pbe_forces.shape
        x, y = pbe_energies / atoms * 27.2114079527, mace_energies / atoms * 27.2114079527
        axes[0].scatter(x, y, label=f'{label} Energies')

        rmse = np.sqrt(np.mean((x - y) ** 2))
        axes[0].text(0.95, 0.10, f'RMSE = {rmse:.4f}', ha='right', va='bottom', transform=axes[0].transAxes, fontsize=12)

        slope, intercept, _, _, _ = scipy.stats.linregress(x, y)
        sns.regplot(x=x, y=y, ax=axes[0], scatter=False, color='r')
        axes[0].text(0.95, 0.05, f'y = {round(intercept, 3)} + {round(slope, 3)}x',
                     ha='right', va='bottom', transform=axes[0].transAxes, fontsize=12)

        #axes[0].set_title(f'Energy Comparison for {label}')
        axes[0].set_xlabel('PBE Energy / atom (eV)')
        axes[0].set_ylabel('MACE Energy / atom (eV)')

        if axis_lims_e is not None:
            axes[0].set_xlim(axis_lims_e[0])
            axes[0].set_ylim(axis_lims_e[1])
            if len(axis_lims_e) > 2:
                axes[0].set_xticks(axis_lims_e[2])
                axes[0].set_yticks(axis_lims_e[3])
    else:
        axes[0].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axes[0].set_title(f'{label} - No Energy Data')

    # Force plot
    if pbe_forces is not None and mace_forces is not None:
        pbe_forces_flat = pbe_forces.reshape(-1) * 51.422086190832  # Convert to eV/Å
        mace_forces_flat = mace_forces.reshape(-1) * 51.422086190832  # Convert to eV/Å
        slope, intercept, _, _, _ = scipy.stats.linregress(pbe_forces_flat, mace_forces_flat)
        rmse = np.sqrt(np.mean((pbe_forces_flat - mace_forces_flat) ** 2))

        axes[1].scatter(pbe_forces_flat, mace_forces_flat, marker='.', alpha=0.5, label=f'{label} Forces')
        sns.regplot(x=pbe_forces_flat, y=mace_forces_flat, ax=axes[1], scatter=False, color='r')
        axes[1].text(0.95, 0.10, f'RMSE = {rmse:.4f}', ha='right', va='bottom', transform=axes[1].transAxes, fontsize=12)
        axes[1].text(0.95, 0.05, f'y = {round(intercept, 3)} + {round(slope, 3)}x',
                ha='right', va='bottom', transform=axes[1].transAxes, fontsize=12)

        #axes[1].set_title(f'Force Comparison for {label}')
        axes[1].set_xlabel('PBE Forces (eV/Å)')
        axes[1].set_ylabel('MACE Forces (eV/Å)')

        if axis_lims_f is not None:
            axes[1].set_xlim(axis_lims_f[0])
            axes[1].set_ylim(axis_lims_f[1])
            if len(axis_lims_f) > 2:
                axes[1].set_xticks(axis_lims_f[2])
                axes[1].set_yticks(axis_lims_f[3])
    else:
        axes[1].text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
        axes[1].set_title(f'{label} - No Data')

    plt.tight_layout()
    plt.savefig(f'Force_Energy_{label}.png')
    plt.show()

def main():
    """
    Main function to generate plots for provided data paths.
    """
    data_paths = [
        '../../Results/silanol/aiMD353',
        '../../Results/silanol/aiMD773_1',
        '../../Results/silanol/vsMD_3'

    ]
    labels = ['aiMD353', 'aiMD773_1', 'vsMD_3']

    # Energy plots
    # plot_energies(data_paths,
    #               labels,
    #               file1='merged_pbe.npz',
    #               file2='mace1-forces.npz',
    #               axis_lims=((-281.12, -281), (-280.815, -280.75)),#, (-281.12, -281.08, -281.04, -281), (-280.78)),
    #               subplot_shape=(1,3)
    #               )
    #
    # # Force plots
    # plot_forces(data_paths,
    #             labels,
    #             file1='merged_pbe.npz',
    #             file2='mace1-forces.npz',
    #             subplot_shape=(1,3)
    #             )

    # Both in one figure
    plot_energy_force_combined(
                data_path='../../Results/silanol/aiMD353',
                label='aiMD353',
                file1='merged_pbe.npz',
                file2='mace1-forces.npz',
                axis_lims_e=(
                  (-281.12, -281.06),
                  (-280.815, -280.78),
                  (-281.12, -281.10, -281.08, -281.06),
                  (-280.81, -280.8, -280.79, -280.78)
                ),
                axis_lims_f=((-8,8),(-8,8))
    )

    plot_energy_force_combined(
                data_path='../../Results/silanol/vsMD_5',
                label='vsMD',
                file1='merged_pbe.npz',
                file2='mace1-forces.npz',
                axis_lims_e=(
                  (-281.12, -281.056),
                  (-280.815, -280.78),
                  (-281.12, -281.10, -281.08, -281.06),
                  (-280.81, -280.8, -280.79, -280.78)
                ),
                axis_lims_f = ((-8, 8.1), (-8, 8))
    )

    plot_energy_force_combined(
                data_path='../../Results/silanol/aiMD773_1',
                label='aiMD773',
                file1='merged_pbe.npz',
                file2='mace1-forces.npz',
                axis_lims_e=(
                  (-281.12, -281),
                  (-280.815, -280.75),
                  (-281.12, -281.08, -281.04, -281),
                  (-280.81, -280.79, -280.77, -280.75)
                ),
                axis_lims_f = ((-15, 15), (-15, 15))
    )

if __name__ == "__main__":
    main()
