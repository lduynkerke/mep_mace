import os
import random
import argparse
from ase.io import read, write
from ase import Atoms
import numpy as np

def save_coord_sys(atoms, output_dir):
    """
    Saves an ASE Atoms object to a coord.sys file format.

    :param atoms: ASE Atoms object to save.
    :type atoms: Atoms
    :param output_dir: Directory to save the coord.sys file.
    :type output_dir: str
    """
    os.makedirs(output_dir, exist_ok=True)
    coord_path = os.path.join(output_dir, 'coord.sys')
    with open(coord_path, 'w') as coord_file:
        coord_file.write("&COORD\n")
        for atom, position in zip(atoms, atoms.positions):
            coord_file.write(f"{atom.symbol} {position[0]} {position[1]} {position[2]}\n")
        coord_file.write("&END COORD\n")

def process_structures(input_file, output_dir, num_structures, rattle_std):
    """
    Reads structures from an XYZ file, selects random structures, applies rattling, 
    and saves the results as .xyz and coord.sys files.

    :param input_file: Path to the input .xyz file.
    :type input_file: str
    :param output_dir: Directory to save the output files.
    :type output_dir: str
    :param num_structures: Number of structures to select randomly.
    :type num_structures: int
    :param rattle_std: Standard deviation for the rattling operation.
    :type rattle_std: float
    """
    structures = read(input_file, index=":")
    num_available = len(structures)
    num_to_select = min(num_structures, num_available)
    selected_indices = random.sample(range(num_available), num_to_select)
    selected_structures = [structures[i] for i in selected_indices]

    for i, structure in enumerate(selected_structures):
        rattled_structure = structure.copy()
        rattled_structure.rattle(stdev=rattle_std, rng=np.random)
        subdir = os.path.join(output_dir, f"conf_{i}")
        os.makedirs(subdir, exist_ok=True)
        write(os.path.join(subdir, "struct.xyz"), rattled_structure)
        write(os.path.join(subdir, "original.xyz"), structure)
        save_coord_sys(rattled_structure, subdir)

def main():
    """
    Parses command-line arguments and initiates processing of structures.
    """
    parser = argparse.ArgumentParser(description="Randomly select and rattle structures from an XYZ file.")
    parser.add_argument("-i", "--input", type=str, required=True, help="Path to the input .xyz file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Directory to save the output.")
    parser.add_argument("-n", "--num_structures", type=int, default=10, help="Number of structures to select randomly.")
    parser.add_argument("-r", "--rattle_std", type=float, default=0.1, help="Standard deviation for rattling.")
    args = parser.parse_args()

    try:
        process_structures(args.input, args.output, args.num_structures, args.rattle_std)
        print(f"Successfully processed {args.num_structures} structures. Outputs saved to {args.output}.")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()

