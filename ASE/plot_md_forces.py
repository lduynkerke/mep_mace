import math
import numpy as np
import matplotlib.pyplot as plt
from ase import units
from ase.io import Trajectory, read, write
from ase.neighborlist import NeighborList
from ase.data import atomic_masses
from collections import defaultdict
from scipy.stats import maxwell
from ase.geometry.analysis import Analysis
from aseMolec import anaAtoms as aa
from statistics import fmean
from itertools import combinations, combinations_with_replacement
import copy
import os

def main():
    mace_forces_file = 'traj/nose_353K.xyz'  # File containing MACE forces
    pbe_forces_file = '../Forces_and_traj_353_1.xyz'  # File containing PBE forces
    mace_r2_forces_file = 'nose_353K_R2.xyz'
    n_atoms = 506  # Number of atoms
    masses = {
        'H': atomic_masses[1],
        'O': atomic_masses[8],
        'Si': atomic_masses[14],
        'C': atomic_masses[6],
        'Zr': atomic_masses[40],
    }

    # Process atoms
    # atoms_list = read('traj/nose_353K.xyz', ':')  # Load all configurations from the file
    # write('traj/nose_353K.traj', atoms_list)

    # Convert
    traj_file = 'traj/nose_353K_25.traj'
    # trajectory = read(traj_file, ':')  # Read all frames from the .traj file

    # Save as .xyz file
    # xyz_file = 'traj/nose_353K_25.xyz'
    # write(xyz_file, trajectory)
    # print(f"Converted {traj_file} to {xyz_file}")

    # Inspect atom object
    # print_traj_attributes('nose_353K.traj')

    # Analyze forces
    plot_std3("traj/nose_353K.xyz", "nose_353K_R2.xyz", 401)
    #analyze_forces(mace_forces_file, pbe_forces_file, n_atoms)
    #analyze_forces(mace_r2_forces_file, pbe_forces_file, n_atoms)

    # Analyze bonds
    # analyze_bonds('nose_353K.traj', cutoff=2.5)
    elements = ['H', 'O', 'C', 'Si', 'Zr']
    # analyze_bonds_full('traj/nose_353K.traj', elements, output_dir="md_bond_analysis")
    # coords = coordination_numbers = calculate_and_plot_coordination_numbers('traj/nose_353K.traj', elements, unique=True)

    #analyze_and_plot('nose_353K.traj', output_dir="analysis_plots")

    # Plot RDF
    # compute_rdf(mace_r2_forces_file)
    # r, g_r = compute_rdf('nose_353K.traj')
    # plt.figure()
    # plt.plot(r, g_r, label='RDF')
    # plt.xlabel('r (Å)')
    # plt.ylabel('g(r)')
    # plt.legend()
    # plt.show()

    # Plot velocity distribution
    # analyze_velocity_distribution('nose_353K.traj', temperature=353, mass=masses)
    # analyze_velocity_distribution_by_species('traj/nose_353K.traj')
    # plot_bond_length_distribution('traj/nose_353K.traj', cutoff=2.5)
    # bond_angle_distribution('nose_353K.traj', cutoff=2.5)
    # plot_coordination_numbers('nose_353K.traj', cutoff=2.5)
    # analyze_time_evolution('nose_353K.traj', cutoff=2.5)

def plot_std(traj1, traj2, steps):
    # Load forces from two different XYZ files
    nose353_1 = load_forces(traj1, force_column_start=10, force_column_end=13)[:steps, :, :]
    nose353_2 = load_forces(traj2, force_column_start=10, force_column_end=13)[:steps, :, :]

    # Diff
    std = np.mean(np.abs(nose353_1 - nose353_2)/np.sqrt(2), axis=(1, 2))
    nose1_mean = np.mean(nose353_1, axis=(1, 2))
    nose2_mean = np.mean(nose353_2, axis=(1, 2))
    mean = np.mean(nose353_1 - nose353_2, axis=(1, 2))
    time_steps = np.arange(np.shape(nose353_1)[0])

    # # Compute force magnitudes for both datasets
    # magnitudes_1 = np.linalg.norm(nose353_1, axis=2)
    # magnitudes_2 = np.linalg.norm(nose353_2, axis=2)
    #
    # # Compute average force magnitude at each timestep
    # avg_force_1 = np.mean(magnitudes_1, axis=1)
    # avg_force_2 = np.mean(magnitudes_2, axis=1)
    # mean_force = (avg_force_1 + avg_force_2) / 2
    #
    # # Compute the standard deviation between the two datasets
    # std_dev = np.std([avg_force_1, avg_force_2], axis=0)
    #
    # # Plotting the results
    # time_steps = np.arange(101)

    plt.figure(figsize=(12, 6))
    plt.plot(time_steps, nose1_mean, label="Trajectory 1", color="C0")
    plt.plot(time_steps, nose2_mean, label="Trajectory 2", color="C1")
    plt.plot(time_steps, mean, label="Mean", color="C2")
    #plt.fill_between(time_steps, mean, mean+std, color="C0", alpha=0.2)
    #plt.fill_between(time_steps, mean-std, mean, color="C1", alpha=0.2)

    plt.title("Average Force Magnitude vs. Time")
    plt.xlabel("Time Steps")
    plt.ylabel("Average Force Magnitude")
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_std3(traj1, traj2, steps):
    nose353_1 = load_forces(traj1, force_column_start=10, force_column_end=13)[:steps, :, :]
    nose353_2 = load_forces(traj2, force_column_start=10, force_column_end=13)[:steps, :, :]

    stacked_forces = np.stack([nose353_1, nose353_2], axis=3)  # Shape: (steps, particles, 3, 2)
    mean_forces = np.mean(stacked_forces, axis=(1, 3))  # Mean across particles and datasets (shape: steps, 3)
    std_dev_forces = np.std(stacked_forces, axis=(1, 3))  # Std dev across particles and datasets (shape: steps, 3)
    avg_force_magnitude = np.linalg.norm(mean_forces, axis=1)
    avg_force_std = np.linalg.norm(std_dev_forces, axis=1)

    mean_x, mean_y, mean_z = mean_forces.T  # Mean values for x, y, z components
    std_x, std_y, std_z = std_dev_forces.T  # Std dev values for x, y, z components
    time_steps = np.arange(steps) * 0.5  # Assuming 1 time step = 0.5 fs

    # Plot magnitude only
    plt.figure(figsize=(8, 6))
    plt.plot(time_steps, avg_force_magnitude, label="Avg Force", color="black")
    plt.fill_between(
        time_steps, avg_force_magnitude - avg_force_std, avg_force_magnitude + avg_force_std,
        color="red", alpha=0.2, label="Std Dev"
    )
    plt.title("Force Magnitude")
    plt.xlabel("Time (fs)")
    plt.ylabel("Force (eV/Å)")
    plt.savefig("std_magnitude.png")
    plt.show()

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Average force magnitude
    axes[0, 0].plot(time_steps, avg_force_magnitude, label="Avg Force", color="purple")
    axes[0, 0].fill_between(
        time_steps, avg_force_magnitude - avg_force_std, avg_force_magnitude + avg_force_std,
        color="purple", alpha=0.2, label="Std Dev"
    )
    axes[0, 0].set_title("Force Magnitude")
    axes[0, 0].set_xlabel("Time (fs)")
    axes[0, 0].set_ylabel("Force (eV/Å)")
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Top right: X-component
    axes[0, 1].plot(time_steps, mean_x, label="Mean X", color="red")
    axes[0, 1].fill_between(
        time_steps, mean_x - std_x, mean_x + std_x, color="red", alpha=0.2, label="Std Dev X"
    )
    axes[0, 1].set_title("X-Component of Force")
    axes[0, 1].set_xlabel("Time (fs)")
    axes[0, 1].set_ylabel("Force (eV/Å)")
    axes[0, 1].legend()
    axes[0, 1].grid(True)

    # Bottom left: Y-component
    axes[1, 0].plot(time_steps, mean_y, label="Mean Y", color="blue")
    axes[1, 0].fill_between(
        time_steps, mean_y - std_y, mean_y + std_y, color="blue", alpha=0.2, label="Std Dev Y"
    )
    axes[1, 0].set_title("Y-Component of Force")
    axes[1, 0].set_xlabel("Time (fs)")
    axes[1, 0].set_ylabel("Force (eV/Å)")
    axes[1, 0].legend()
    axes[1, 0].grid(True)

    # Bottom right: Z-component
    axes[1, 1].plot(time_steps, mean_z, label="Mean Z", color="green")
    axes[1, 1].fill_between(
        time_steps, mean_z - std_z, mean_z + std_z, color="green", alpha=0.2, label="Std Dev Z"
    )
    axes[1, 1].set_title("Z-Component of Force")
    axes[1, 1].set_xlabel("Time (fs)")
    axes[1, 1].set_ylabel("Force (eV/Å)")
    axes[1, 1].legend()
    axes[1, 1].grid(True)

    plt.tight_layout()
    plt.savefig('std_plot.png')
    plt.show()

def print_traj_attributes(traj_file):
    """
    Reads a .traj file and prints all attributes of the Atoms objects in it.
    :param traj_file: Path to the .traj file.
    """
    traj = Trajectory(traj_file)  # Open the trajectory file
    print(f"Trajectory contains {len(traj)} frames.")

    for i, atoms in enumerate(traj):
        print(f"\n--- Frame {i + 1} ---")
        print(f"Number of atoms: {len(atoms)}")
        print(f"Atomic symbols: {atoms.get_chemical_symbols()}")
        print(f"Positions (Å):\n{atoms.get_positions()}")

        if atoms.has('momenta'):
            print(f"Momenta (amu * Å/fs):\n{atoms.get_momenta()}")
            print(f"Velocities (Å/fs):\n{atoms.get_velocities()}")

        if atoms.has('forces'):
            print(f"Forces (eV/Å):\n{atoms.get_forces()}")

        if atoms.has('charges'):
            print(f"Charges:\n{atoms.get_charges()}")

        print(f"Potential Energy (eV): {atoms.get_potential_energy()}")

        try:
            kinetic_energy = atoms.get_kinetic_energy()
            print(f"Kinetic Energy (eV): {kinetic_energy}")
        except AttributeError:
            print("Kinetic energy not available (no velocities).")

        total_energy = atoms.get_total_energy()
        print(f"Total Energy (eV): {total_energy}")

def load_forces(filename, force_column_start, force_column_end):
    """
    Load forces from the provided XYZ file, skipping header lines in each block.
    Parameters:
        filename (str): Path to the XYZ file.
        force_column_start (int): Starting index for the force columns.
        force_column_end (int): Ending index for the force columns (exclusive).
    Returns:
        forces (numpy.ndarray): Array of forces, shape (n_timesteps, n_atoms, 3).
    """
    forces = []
    with open(filename, 'r') as file:
        lines = file.readlines()
        i = 0
        while i < len(lines):
            if lines[i].strip().isdigit():
                n_atoms = int(lines[i].strip())
                i += 2  # Skip the first two lines (digit line and info line)

                block_forces = []
                for _ in range(n_atoms):
                    if i < len(lines) and lines[i].strip():
                        data = lines[i].split()
                        force = list(map(float, data[force_column_start:force_column_end]))
                        block_forces.append(force)
                    i += 1
                forces.append(block_forces)
            else:
                i += 1  # Increment if unexpected line
    forces = np.array(forces)
    return forces

def analyze_forces(mace_forces_file, pbe_forces_file, n_atoms):
    """
    Analyze and compare MACE and PBE forces.
    Parameters:
        mace_forces_file (str): Path to the MACE XYZ file.
        pbe_forces_file (str): Path to the PBE XYZ file.
        n_atoms (int): Number of atoms in the system.
    """
    mace_forces = load_forces(mace_forces_file, force_column_start=10, force_column_end=13)[:10001,:,:].flatten()
    nose773 = load_forces('traj/nose_773K.xyz', force_column_start=10, force_column_end=12)[:10001,:,:].flatten()
    nose353 = load_forces('traj/nose_353K.xyz', force_column_start=10, force_column_end=12)[:10001,:,:].flatten()
    pbe_forces = load_forces(pbe_forces_file, force_column_start=4, force_column_end=6).flatten()
    print(np.max(mace_forces), np.min(mace_forces))
    print(np.argmax(mace_forces), np.argmin(mace_forces))
    print(np.argmax(mace_forces) // (506 * 3), np.argmax(mace_forces) % (506 * 3) // 3, np.argmax(mace_forces) % 3)
    print(np.argmin(mace_forces) // (506 * 3), np.argmin(mace_forces) % (506 * 3) // 3, np.argmin(mace_forces) % 3)

    # Print statistics
    mace_mean = np.mean(mace_forces)
    mace_variance = np.var(mace_forces)
    pbe_mean = np.mean(pbe_forces)
    pbe_variance = np.var(pbe_forces)

    print("MACE Forces:")
    print(f"Mean: {mace_mean}, Variance: {mace_variance}")
    print("PBE Forces:")
    print(f"Mean: {pbe_mean}, Variance: {pbe_variance}")

    # Histogram of force magnitudes
    # plt.figure()
    # plt.hist(np.log10(pbe_forces), bins=250, alpha=0.7, color='orange', label='MACE')
    # plt.hist(np.log10(nose353), bins=250, alpha=0.7, color='blue', label='PBE')
    # plt.title("Histogram of Force Magnitudes")
    # plt.xlabel("Force Magnitude (eV/Å)")
    # plt.ylabel("Frequency")
    # plt.legend()
    # plt.savefig("force_magnitudes_histogram.png")
    # plt.show()

    # Box plot for both datasets
    plt.figure()
    # plt.boxplot([pbe_forces, nose353, nose773], labels=['PBE Forces', 'MACE Forces 353K', 'MACE Forces 773K'])
    plt.boxplot([pbe_forces, nose353], labels=['PBE Forces', 'MACE Forces 353K'])
    plt.title("Box Plot of Forces")
    plt.ylabel("Forces (eV/A)")
    plt.savefig("force_box.png")
    plt.show()

def compute_rdf(traj_name, r_max=10.0, n_bins=100):
    traj = read(traj_name, '50:') #ignore first 50 frames
    for at in traj:
        at.pbc = True #create a fake box for rdf compatibility
        at.cell = [100,100,100]
    rdf = aa.compute_rdfs_traj_avg(traj, rmax=5, nbins=70) #aseMolec provides functionality to compute RDFs
    plt.plot(rdf[1], rdf[0]['OSi_intra'], '.-', alpha=0.7, linewidth=3)

    # rdf = aa.compute_rdfs_traj_avg(traj, rmax=5, nbins=70) #aseMolec provides functionality to compute RDFs
    #
    # all_rdfs = []
    # for atoms in traj:
    #     rdf, r = radial_distribution_function(atoms.positions, r_max, n_bins)
    #     all_rdfs.append(rdf)
    # avg_rdf = np.mean(all_rdfs, axis=0)
    #
    # plt.plot(r, avg_rdf)
    # plt.xlabel('r (Å)')
    # plt.ylabel('g(r)')
    # plt.title('Radial Distribution Function')
    # plt.show()

def analyze_velocity_distribution_by_species(traj_name):
    traj = Trajectory(traj_name)
    species_speeds = defaultdict(list)

    for atoms in traj:
        velocities = atoms.get_velocities()
        speeds = np.linalg.norm(velocities, axis=1)  # Calculate magnitudes of velocities
        for atom, speed in zip(atoms, speeds):
            species_speeds[atom.symbol].append(speed)

    for element, speeds in species_speeds.items():
        loc, scale = maxwell.fit(speeds)  # Maxwell-Boltzmann parameters loc and scale
        x = np.linspace(min(speeds), max(speeds), 500)
        pdf = maxwell.pdf(x, loc=loc, scale=scale)

        plt.hist(speeds, bins=200, density=True, label=f'Simulation {element}')
        plt.plot(x, pdf, 'r-', label=f'Maxwell-Boltzmann {element}')
        plt.xlabel('Speed (m/s)')
        plt.ylabel('Probability Density')
        plt.legend(loc='upper right')
        plt.savefig(f'velocity_fit_{element}.png')
        plt.show()

def bond_angle_distribution(traj_name, cutoff=2.0):
    traj = Trajectory(traj_name)
    bond_angles = []

    for atoms in traj:
        cutoffs = [cutoff] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        for i in range(len(atoms)):
            neighbors, offsets = nl.get_neighbors(i)
            for j, offset_j in zip(neighbors, offsets):
                for k, offset_k in zip(neighbors, offsets):
                    if j < k:
                        vec1 = atoms.positions[j] - atoms.positions[i]
                        vec2 = atoms.positions[k] - atoms.positions[i]
                        vec1 /= np.linalg.norm(vec1)
                        vec2 /= np.linalg.norm(vec2)
                        angle = math.degrees(math.acos(np.dot(vec1, vec2)))
                        bond_angles.append(angle)

    plt.hist(bond_angles, bins=200, range=(0, 180), color='C3', alpha=0.7)
    plt.xlabel('Bond Angle (°)')
    plt.ylabel('Frequency')
    plt.title('Bond Angle Distribution')
    plt.savefig('bond_angle_dist.png')
    plt.show()

def analyze_time_evolution(traj_name, cutoff=2.0):
    traj = Trajectory(traj_name)
    bond_lengths_over_time = []
    coord_numbers_over_time = []

    for atoms in traj:
        bond_lengths = []
        coord_numbers = []

        cutoffs = [cutoff] * len(atoms)
        nl = NeighborList(cutoffs, self_interaction=False, bothways=True)
        nl.update(atoms)

        for i in range(len(atoms)):
            neighbors, _ = nl.get_neighbors(i)
            coord_numbers.append(len(neighbors))
            for j in neighbors:
                bond_lengths.append(atoms.get_distance(i, j, mic=True))

        bond_lengths_over_time.append(np.mean(bond_lengths))
        coord_numbers_over_time.append(np.mean(coord_numbers))

    plt.plot(bond_lengths_over_time, label='Average Bond Length')
    plt.plot(coord_numbers_over_time, label='Average Coordination Number')
    plt.xlabel('Frame')
    plt.ylabel('Value')
    plt.title('Time Evolution')
    plt.legend()
    plt.savefig('time_evolution.png')
    plt.show()

def analyze_and_plot(traj_file, output_dir="analysis_plots"):
    """
    Performs multiple analyses on an MD trajectory file and generates plots for each analysis.

    :param traj_file: Path to the trajectory file (e.g., .traj, .xyz).
    :param output_dir: Directory to save the analysis plots. Defaults to "analysis_plots".
    """
    import os
    os.makedirs(output_dir, exist_ok=True)

    # Load trajectory
    print(f"Loading trajectory from {traj_file}...")
    traj = read(traj_file, ':')  # Load all frames
    analysis = Analysis(traj)

    print("Performing analysis and plotting results...")

    # 3. Radial Distribution Function (RDF)
    # r, g_r = analysis.get_rdf(rmax=2.5)
    # plt.figure()
    # plt.plot(r, g_r, color='blue')
    # plt.title("Radial Distribution Function")
    # plt.xlabel("Distance (Å)")
    # plt.ylabel("g(r)")
    # plt.savefig(f"{output_dir}/rdf.png")
    # plt.close()
    #
    # # 4. Angle Distribution
    # angles, counts = analysis.get_angles()
    # plt.figure()
    # plt.bar(angles, counts, width=2, color='orange', edgecolor='black')
    # plt.title("Angle Distribution")
    # plt.xlabel("Angle (degrees)")
    # plt.ylabel("Frequency")
    # plt.savefig(f"{output_dir}/angle_distribution.png")
    # plt.close()
    # print(f"All analyses complete. Plots saved in '{output_dir}'.")

def analyze_bonds_full(traj_file, elements, output_dir="analysis_plots", unique=True):
    """
    Analyzes bond lengths across specified elements in an MD trajectory.

    :param traj_file: Path to the trajectory file (e.g., .traj, .xyz).
    :param elements: List of element symbols (e.g., ['H', 'O', 'C', 'Si', 'Zr']).
    :param output_dir: Directory to save the analysis plots. Defaults to "analysis_plots".
    :param unique: Whether to count only unique bonds (A-B and not B-A separately). Defaults to True.
    """
    bond_histograms = {comb: [] for comb in combinations_with_replacement(elements, 2)}
    bond_avg_vs_time = {comb: [] for comb in combinations_with_replacement(elements, 2)}
    os.makedirs(output_dir, exist_ok=True)
    traj = read(traj_file, ':')
    print(f"Loading trajectory from {traj_file}...")
    print(elements)

    for struct in traj[:10000]:
        analysis = Analysis(struct)
        for (element_A, element_B) in combinations_with_replacement(elements, 2):
            #print(element_A, element_B)
            bond_lengths = []
            frame_bonds = analysis.get_bonds(element_A, element_B, unique=unique)[0]
            for xx, yy in frame_bonds:
                length = np.linalg.norm(struct.positions[xx] - struct.positions[yy])
                bond_lengths.append(length)
            if bond_lengths:
                bond_histograms[(element_A, element_B)].extend(bond_lengths)
                bond_avg_vs_time[(element_A, element_B)].append(fmean(bond_lengths))

    print("Generating bond length histograms...")
    for element_A in elements:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True, sharex=True)
        fig.suptitle(f"Bond Length Histograms for {element_A}")
        axes = axes.flatten()
        pair_idx = 0

        for element_B in elements:
            if (element_A, element_B) in combinations_with_replacement(elements, 2):
                bond_lengths = bond_histograms[(element_A, element_B)]
            elif (element_B, element_A) in combinations_with_replacement(elements, 2):
                bond_lengths = bond_histograms[(element_B, element_A)]
            else:
                continue

            if pair_idx >= len(axes) or not bond_lengths:  # Ensure we only plot up to 4 pairs
                break

            ax = axes[pair_idx]
            ax.hist(bond_lengths, bins=30, range=(0, 5), color='skyblue', edgecolor='black')
            ax.set_title(f"{element_A}-{element_B}")
            ax.set_xlabel("Bond Length (Å)")
            ax.set_ylabel("Frequency")
            ax.grid(True, linestyle="--", alpha=0.7)
            pair_idx += 1

        plt.tight_layout()
        plt.savefig(f"{output_dir}/bond_length_histograms_{element_A}.png")
        plt.close()

    print("Plot specific bonds...")
    bonds_to_plot = {
        ("Si", "O"): "red",
        ("C", "H"): "green",
        ("O", "H"): "blue",
        ("Zr", "C"): "orange"
    }

    plt.figure(figsize=(8, 6))
    for (element_A, element_B), color in bonds_to_plot.items():
        if (element_A, element_B) in combinations_with_replacement(elements, 2):
            avg_bond_lengths = bond_avg_vs_time[(element_A, element_B)]
        elif (element_B, element_A) in combinations_with_replacement(elements, 2):
            avg_bond_lengths = bond_avg_vs_time[(element_B, element_A)]
        else:
            print(f"Warning: Bond {element_A}-{element_B} not found in data.")
            continue

        plt.plot(range(len(avg_bond_lengths)), avg_bond_lengths, label=f"{element_A}-{element_B}")#, color=color)

    plt.title("Bond Lengths vs Time for Selected Bonds")
    plt.xlabel("Time Step")
    plt.ylabel("Average Bond Length (\u00c5)")
    plt.legend()
    #plt.grid(True, linestyle="--", alpha=0.7)
    plt.tight_layout()

    plt.savefig(f"{output_dir}/selected_bond_lengths_vs_time.png")
    plt.close()
    print("Plot saved as selected_bond_lengths_vs_time.png.")

    print("Generating average bond length vs time plots...")
    for element_A in elements:
        fig, axes = plt.subplots(2, 2, figsize=(10, 8), sharey=True, sharex=True)
        fig.suptitle(f"Average Bond Length vs Time for {element_A}")
        axes = axes.flatten()
        pair_idx = 0

        for element_B in elements:
            if (element_A, element_B) in combinations_with_replacement(elements, 2):
                avg_bond_lengths = bond_avg_vs_time[(element_A, element_B)]
            elif (element_B, element_A) in combinations_with_replacement(elements, 2):
                 avg_bond_lengths = bond_avg_vs_time[(element_B, element_A)]
            else:
                continue
            if pair_idx >= len(axes) or not avg_bond_lengths:  # Ensure we only plot up to 4 pairs
                break

            ax = axes[pair_idx]
            ax.plot(range(len(avg_bond_lengths)), avg_bond_lengths, color='blue')
            ax.set_title(f"{element_A}-{element_B}")
            ax.set_xlabel("Time Step")
            ax.set_ylabel("Average Bond Length (Å)")
            ax.grid(True, linestyle="--", alpha=0.7)
            pair_idx += 1

        plt.tight_layout()
        plt.savefig(f"{output_dir}/avg_bond_length_vs_time_{element_A}.png")
        plt.close()

    print(f"All plots saved in '{output_dir}'.")

def calculate_and_plot_coordination_numbers(traj_file, elements, unique=True):
    """
    Calculate and plot histograms of coordination numbers for every element.

    Parameters:
    - traj: ase.io.Trajectory object containing the MD trajectory.
    - analysis: ase.geometry.analysis.Analysis object for bond analysis.
    - elements: List of element symbols (e.g., ['H', 'O', 'C', 'Si', 'Zr']).
    - unique: Whether to count unique bonds (A-B only, not B-A).

    Returns:
    - coordination_numbers: Dict with element keys and lists of coordination numbers.
    """
    print(f"Loading trajectory from {traj_file}...")
    traj = read(traj_file, ':')  # Load all frames
    coordination_numbers = {element: [] for element in elements}

    # Init empty lists
    struct0 = traj[0]
    atoms_by_element = {element: [] for element in elements}
    bond_counts = {atom_idx: 0 for atom_idx in range(len(struct0))}
    for atom_idx, atom in enumerate(struct0):
        if atom.symbol in atoms_by_element:
            atoms_by_element[atom.symbol].append(atom_idx)

    # Loop over all elements
    for struct_idx, struct in enumerate(traj):
        analysis = Analysis(struct)
        atom_bond_counts = copy.deepcopy(bond_counts)
        for (element_A, element_B) in combinations_with_replacement(elements, 2):
                frame_bonds = analysis.get_bonds(element_A, element_B, unique=unique)[0]
                for xx, yy in frame_bonds:
                    atom_bond_counts[xx] += 1
                    atom_bond_counts[yy] += 1

        for element, atom_indices in atoms_by_element.items():
            coordination_numbers[element].extend(atom_bond_counts[atom_idx] for atom_idx in atom_indices)

    # Plot
    for i, element in enumerate(elements):
        plt.figure()
        labels, counts = np.unique(coordination_numbers[element], return_counts=True)
        plt.bar(labels, counts, align='center')
        plt.title(f"Coordination Numbers for {element}")
        plt.xlabel("Coordination Number")
        plt.ylabel("Frequency")
        plt.xticks(np.arange(min(np.min(labels), 1), np.max(labels)+1, 1).tolist())
        plt.tight_layout()
        plt.savefig('coordnum_'+element+'.png')
        plt.show()
        print(element)
        print(labels)
        print(counts)

    return coordination_numbers

if __name__ == "__main__":
    main()
