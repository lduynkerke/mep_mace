import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_npz_data(filepath, energy_key='energies', force_key='forces'):
    """Load energy and force data from an NPZ file."""
    data = np.load(filepath)
    return data[energy_key], data[force_key]

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

def compute_errors(pbe_data, mace_data):
    """
    Compute the absolute errors between PBE and MACE data for both energies and forces,
    excluding cases where MACE energies or forces are exactly 0.0.

    :param pbe_data: A tuple containing PBE energies and PBE forces.
    :type pbe_data: tuple
    :param mace_data: A tuple containing MACE energies and MACE forces.
    :type mace_data: tuple
    :return: A tuple containing the energy and force errors.
    :rtype: tuple
    """
    pbe_energies, pbe_forces = pbe_data
    mace_energies, mace_forces = mace_data

    if mace_energies is not None and pbe_energies is not None:
        valid_energy_mask = mace_energies != 0.0
        energy_diff = np.abs(pbe_energies[valid_energy_mask] - mace_energies[valid_energy_mask])
    else:
        energy_diff = None

    if mace_forces is not None and pbe_forces is not None:
        pbe_forces = pbe_forces.reshape(-1)
        mace_forces = mace_forces.reshape(-1)
        valid_force_mask = mace_forces != 0.0
        force_diff = np.abs(pbe_forces[valid_force_mask] - mace_forces[valid_force_mask])
    else:
        force_diff = None

    return energy_diff, force_diff

def process_and_append_errors(pbe_energies, pbe_forces, mace_energies_raw, mace_forces_raw, energy_errors, force_errors, dft=False):
    """Process raw MACE energies and forces, compute errors, and append to lists."""
    mace_forces = reshape_forces(mace_forces_raw) * 51.422086190832  # Scale forces
    mace_energies = mace_energies_raw / mace_forces.shape[0] * 27.2114079527  # Scale energies

    if dft:
        mace_energies = mace_energies + np.mean(pbe_energies) - np.mean(mace_energies)

    # Compute errors
    energy_err, force_err = compute_errors(
        (pbe_energies, pbe_forces),
        (mace_energies, mace_forces)
    )
    filtered_force_err = np.log10([err for err in force_err if abs(err) >= 1e-3])
    energy_errors.append(energy_err)
    force_errors.append(filtered_force_err)

def main():
    # Paths and file names
    base_path = '../../Results/av_strain_surface/Adsorption_Complex_Simulations/Velocity_softening_dynamics/'  # Adjust as needed
    pbe_file = base_path + "PBE_forces.npz"
    data_file = base_path + "data.npz"
    mace_files = [
        "mace1_2_forces.npz",
        "mace1_1_100_forces.npz",
        "mace1_1_25_forces.npz",
        "mace1_12_forces.npz",
        "mace1_24_forces.npz",
        "mace3_forces.npz",
        "mace1_forces_random1.npz",
        "mace2_forces.npz",
        "mace_large_forces.npz"
    ]

    # Load and preprocess PBE data
    pbe_energies_raw, pbe_forces_raw = load_npz_data(pbe_file)
    pbe_forces = reshape_forces(pbe_forces_raw) * 51.422086190832  # Scale forces
    pbe_energies = pbe_energies_raw / pbe_forces.shape[0] * 27.2114079527  # Scale energies

    # Initialize error lists
    energy_errors = []
    force_errors = []

    # Process MACE files
    for mace_file in mace_files:
        mace_energies_raw, mace_forces_raw = load_npz_data(base_path + mace_file)
        process_and_append_errors(pbe_energies, pbe_forces, mace_energies_raw, mace_forces_raw, energy_errors,
                                  force_errors)

    # Process data.npz for LDA and BLYP
    data_energies_lda, data_forces_lda = load_npz_data(data_file, energy_key='lda_e', force_key='lda_f')
    data_energies_blyp, data_forces_blyp = load_npz_data(data_file, energy_key='blyp_e', force_key='blyp_f')

    # Append errors for LDA
    process_and_append_errors(pbe_energies, pbe_forces, data_energies_lda, data_forces_lda, energy_errors, force_errors, dft=True)

    # Append errors for BLYP
    process_and_append_errors(pbe_energies, pbe_forces, data_energies_blyp, data_forces_blyp, energy_errors,
                              force_errors, dft=True)

    # Labels for the boxplots
    labels = ["1 \n 250", "1 \n 100", "1 \n 25", "12", "24", "mace1", "random", "epoch \n 250", "large", "LDA", "BLYP"]

    # Plot energy errors
    plt.figure()
    plt.boxplot(energy_errors, patch_artist=True, labels=labels, showfliers=False)
    plt.xticks(rotation=45)
    plt.ylabel('Energy Errors (eV)')
    #plt.ylim(0, 0.25)
    #plt.title('Energy Errors Comparison')
    plt.tight_layout()
    plt.savefig("energy_errors_boxplot.png")
    plt.show()

    # Plot force errors
    plt.figure()
    #plt.boxplot(force_errors, patch_artist=True, labels=labels, showfliers=True)
    sns.violinplot(data=force_errors)
    plt.xticks(range(0, 11), labels=[f'{s}' for s in labels], rotation=45)
    #plt.xticks(rotation=45)
    plt.yticks([-3, -2, -1, 0, 1])
    plt.ylabel('10 log Force Errors (eV/Å)')
    #plt.title('Force Errors Comparison')
    plt.tight_layout()
    plt.savefig("force_errors_boxplot.png")
    plt.show()


# Entry point
if __name__ == "__main__":
    main()