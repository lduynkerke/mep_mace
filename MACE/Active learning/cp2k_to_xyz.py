import os
import argparse

def parse_atomic_numbers(atomic_numbers_str):
    """
    Parses the atomic numbers string into a dictionary.

    :param atomic_numbers_str: String of atomic mappings in the format "H:1,O:8,Si:14".
    :type atomic_numbers_str: str
    :return: Dictionary mapping atom species to their atomic numbers.
    :rtype: dict[str, int]
    """
    atomic_numbers = {}
    for item in atomic_numbers_str.split(","):
        species, number = item.split(":")
        atomic_numbers[species.strip()] = int(number.strip())
    return atomic_numbers

def extract_energy(file_path):
    """
    Extracts the total energy from the output file.

    :param file_path: Path to the energy file.
    :type file_path: str
    :return: Extracted energy value or None if missing.
    :rtype: float | None
    """
    if not os.path.exists(file_path):
        return None  # Missing file
    with open(file_path, 'r') as f:
        for line in f:
            if 'energy [a.u.]' in line:
                return float(line.split()[-1])
    return None  # Default if energy line not found

def parse_coords(file_path):
    """
    Parses coordinates from coords.sys.

    :param file_path: Path to the coordinates file.
    :type file_path: str
    :return: List of atomic coordinates.
    :rtype: list[tuple[str, float, float, float]]
    """
    coords = []
    with open(file_path, 'r') as f:
        lines = f.readlines()
        start = False
        for line in lines:
            if "&COORD" in line:
                start = True
                continue
            if "&END COORD" in line:
                break
            if start:
                parts = line.split()
                coords.append((parts[0], float(parts[1]), float(parts[2]), float(parts[3])))
    return coords

def parse_forces(file_path, atom_count):
    """
    Parses forces from PBE-frc-1_0.xyz, stopping after the specified atom count.

    :param file_path: Path to the forces file.
    :type file_path: str
    :param atom_count: Number of atoms to read.
    :type atom_count: int
    :return: List of force components.
    :rtype: list[tuple[float, float, float]] | None
    """
    forces = []
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()[4:]  # Skip the header lines
            for i, line in enumerate(lines):
                if i >= atom_count:  # Stop when reaching the atom count
                    break
                parts = line.split()
                try:
                    forces.append((float(parts[3]), float(parts[4]), float(parts[5])))
                except (ValueError, IndexError):
                    print(f"Read error in line: {line}")
                    print(f"In file: {file_path}")
                    return None
    except UnicodeDecodeError:
        print(f"Removed corrupt file {file_path}")
        os.remove(file_path)
        return None
    return forces

def write_pbe_ext(output_file, coords, forces, tot_energy, time_step, lattice, atomic_numbers, pbc="T T T"):
    """
    Writes PBE-ext.xyz file with energy, forces, and coordinates.

    :param output_file: Output file path.
    :param coords: List of atomic coordinates.
    :param forces: List of forces.
    :param tot_energy: Total energy of the configuration.
    :param time_step: Time step.
    :param lattice: Lattice information.
    :param atomic_numbers: Dictionary mapping atomic species to atomic numbers.
    :param pbc: Periodic boundary conditions (default: "T T T").
    """
    atom_count = len(coords)
    with open(output_file, 'w') as out:
        out.write(f"{atom_count}\n")
        out.write(
            f"time={time_step:.3f} TotEnergy={tot_energy:.10f} "
            f'cutoff=-1.00000000 nneightol=1.20000000 pbc="{pbc}" '
            f'Lattice="{lattice}" '
            'Properties=species:S:1:pos:R:3:force:R:3:Z:I:1\n'
        )
        for coord, force in zip(coords, forces):
            species, x, y, z = coord
            fx, fy, fz = force
            atomic_number = atomic_numbers.get(species, 0)  # Default to 0 if species not found
            out.write(f"{species} {x:.10f} {y:.10f} {z:.10f} {fx:.10f} {fy:.10f} {fz:.10f} {atomic_number}\n")

def process_folders(root_folder, base_name, start, end, increment, lattice, atom_count, atomic_numbers):
    """
    Processes specified range of configuration folders.

    :param root_folder: Path to the root folder.
    :param base_name: Base name for the folders.
    :param start: Start index of the folders.
    :param end: End index of the folders.
    :param increment: Increment for folder numbers.
    :param lattice: Lattice information.
    :param atom_count: Number of atoms to read in forces.
    :param atomic_numbers: Dictionary mapping atomic species to atomic numbers.
    """
    merged_data = []
    for i in range(start, end + 1, increment):
        folder = f"{base_name}{i}"
        print(f"Processing folder: {folder}")
        folder_path = os.path.join(root_folder, folder)
        if not os.path.isdir(folder_path):
            print(f"Skipping {folder}: Not a directory.")
            continue

        coords_file = os.path.join(folder_path, "coord.sys")
        forces_file = os.path.join(folder_path, "PBE-forces-1_0.xyz")
        energy_file = os.path.join(folder_path, "pbe.out")

        if not (os.path.exists(coords_file) and os.path.exists(forces_file)):
            continue

        coords = parse_coords(coords_file)
        forces = parse_forces(forces_file, atom_count)
        tot_energy = extract_energy(energy_file)

        if not tot_energy or not forces or not coords:
            continue

        time_step = 0.5 * int(i)
        pbe_ext_file = os.path.join(folder_path, "pbe.xyz")
        write_pbe_ext(pbe_ext_file, coords, forces, tot_energy, time_step, lattice, atomic_numbers)

        with open(pbe_ext_file, 'r') as f:
            merged_data.append(f.read())

    merged_file = os.path.join(root_folder, "merged.xyz")
    with open(merged_file, 'w') as out:
        out.write("".join(merged_data))

def main():
    """
    Main function to handle argument parsing and initiate processing.
    """
    parser = argparse.ArgumentParser(
        description="Process configuration folders and generate PBE-ext and merged.xyz files."
    )
    parser.add_argument("--root", type=str, required=True, help="Root folder containing configuration folders.")
    parser.add_argument("--base", type=str, required=True, help="Base name of configuration folders (e.g., 'conf_').")
    parser.add_argument("--range", type=int, nargs=3, required=True, metavar=('START', 'END', 'INTERVAL'),
                        help="Range of folder indices (inclusive).")
    parser.add_argument("--lattice", type=str, required=True, help="Lattice parameters for PBE-ext file.")
    parser.add_argument("--atom_count", type=int, required=True, help="Number of atoms to read in forces files.")
    parser.add_argument("--atomic_numbers", type=str, required=True,
                        help="Atomic numbers as a string, e.g., 'H:1,O:8,Si:14,C:6,Zr:40'.")

    args = parser.parse_args()
    atomic_numbers = parse_atomic_numbers(args.atomic_numbers)

    process_folders(args.root, args.base, args.range[0], args.range[1], args.range[2],
                    args.lattice, args.atom_count, atomic_numbers)

if __name__ == "__main__":
    main()

