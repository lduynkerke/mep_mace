import pandas as pd
from ase.io import read
from mace.calculators import MACECalculator
import re
import io
import argparse
from mpi4py import MPI


def extract_energy_structure(file_input):
    """
        Extracts energies and molecular structures from an XYZ file.

        This function reads an XYZ file, extracts total energies labeled by 'TotEnergy',
        and captures atomic structures associated with those energies.

        :param file_input: The path to the input XYZ file.
        :type file_input: str
        :return: A tuple containing two lists: the extracted energies and structures.
        :rtype: tuple[list[float], list[str]]
        """
    energies = []
    structures = []

    current_structure = []
    reading_structure = False

    with open(file_input, 'r') as f:
        for line in f:
            if 'TotEnergy' in line:
                energy_match = re.search(r"TotEnergy = ([\d.-]+)", line)

                if energy_match:
                    energy = float(energy_match.group(1))
                    energies.append(energy)

                if reading_structure:
                    current_structure.append(line)

            # New structure starts with 506; save the previous structure and start reading the new block
            elif line.strip() == '506':
                if reading_structure and current_structure:
                    structure_block = ''.join(current_structure)
                    structures.append(structure_block)

                reading_structure = True
                current_structure = [line]

            elif reading_structure:
                current_structure.append(line)

    return energies, structures


def process_xyz_file_parallel(file_input, calculator):
    """
    Processes an XYZ file in parallel using MPI, calculates energies for each structure, and compares them.

    The function divides the structures across multiple MPI ranks, assigns each rank a portion of the structures,
    and calculates the predicted energy using a given calculator. The energy difference between the predicted
    and extracted energies is then calculated.

    :param file_input: The path to the input XYZ file.
    :type file_input: str
    :param calculator: A calculator object to compute the potential energy.
    :type calculator: MACECalculator
    :return: A list of dictionaries with energy comparison results, or None for non-root ranks.
    :rtype: list[dict] or None
    """
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()  # Get the rank of the current process
    size = comm.Get_size()  # Get the total number of processes

    extracted_energies, structures = extract_energy_structure(file_input)

    # Divide the structures among the processes
    structures_per_rank = len(structures) // size
    extra_structures = len(structures) % size

    if rank < extra_structures:
        # Distribute the extra structures among the first `extra_structures` ranks
        local_structures = structures[rank * (structures_per_rank + 1):(rank + 1) * (structures_per_rank + 1)]
        local_energies = extracted_energies[rank * (structures_per_rank + 1):(rank + 1) * (structures_per_rank + 1)]
    else:
        # Remaining ranks get an even distribution of structures
        offset = extra_structures * (structures_per_rank + 1)
        local_structures = structures[offset + (rank - extra_structures) * structures_per_rank:
                                      offset + (rank - extra_structures + 1) * structures_per_rank]
        local_energies = extracted_energies[offset + (rank - extra_structures) * structures_per_rank:
                                            offset + (rank - extra_structures + 1) * structures_per_rank]

    local_data = []

    # Process local structures assigned to each rank
    for i, structure_block in enumerate(local_structures):
        structure_io = io.StringIO(structure_block)
        ase_structure = read(structure_io, format='xyz')
        ase_structure.calc = calculator

        predicted_energy = ase_structure.get_potential_energy()
        energy_diff = predicted_energy - local_energies[i]
        print(f"Rank {rank}, structure {i}: predicted_energy={predicted_energy}, energy_diff={energy_diff}")

        local_data.append({
            'Structure': i + 1 + (rank * structures_per_rank),  # Keep track of structure index across all ranks
            'File': file_input,
            'Extracted Energy': local_energies[i],
            'Predicted Energy': predicted_energy,
            'Energy Difference': energy_diff
        })

    # Gather data from all ranks to the root rank (rank 0)
    all_data = comm.gather(local_data, root=0)

    # On the root rank, flatten the gathered data
    if rank == 0:
        all_data = [item for sublist in all_data for item in sublist]
        return all_data
    else:
        return None

def main():
    """
    The main entry point for processing multiple XYZ files using a trained MACE model.

    This function uses MPI for parallel processing and compares energies for each structure in the files.
    The results are saved in a CSV file.

    :return: None
    """
    parser = argparse.ArgumentParser(description="Process multiple XYZ files and compare energies.")
    parser.add_argument('files', nargs='+', help='Input XYZ files')
    parser.add_argument('--model_path', required=True, help='Path to the trained MACE model')
    parser.add_argument('--output_csv', default='energy_differences.csv', help='Output CSV file')

    args = parser.parse_args()

    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    calculator = MACECalculator(model_paths=args.model_path, device='cpu')
    all_data = []

    # Only the root rank will write the results to the CSV file
    for file_input in args.files:
        data = process_xyz_file_parallel(file_input, calculator)

        if rank == 0:
            all_data.extend(data)

    if rank == 0:
        df = pd.DataFrame(all_data)
        df.to_csv(args.output_csv, index=False)
        print(f"Results saved to {args.output_csv}")


if __name__ == '__main__':
    main()