import os
import argparse

def save_last_structures(file_path, output_folder, num_structures=10):
    """
    Extracts the last N molecular structures from an XYZ file and saves them to separate directories.

    This function reads an XYZ file, identifies molecular structures, and saves the last `num_structures` to
    subdirectories named coord_0, coord_1, ..., coord_N-1 inside the specified output folder.

    :param file_path: The path to the input XYZ file.
    :type file_path: str
    :param output_folder: The path to the output directory where results will be saved.
    :type output_folder: str
    :param num_structures: The number of most recent structures to save.
    :type num_structures: int
    """
    try:
        with open(file_path, 'r') as f:
            lines = f.readlines()

        structures = []
        current_structure = []

        for line in lines:
            if line.strip().isdigit():
                if current_structure:
                    structures.append(current_structure)
                    if len(structures) > num_structures:
                        structures.pop(0)
                    current_structure = []
                continue
            
            if line.startswith('Lattice'):
                continue

            elif len(line.split()) >= 4:
                parts = line.split()
                atom_line = f"{parts[0]} {parts[1]} {parts[2]} {parts[3]}\n"
                current_structure.append(atom_line)

        if current_structure:
            structures.append(current_structure)
            if len(structures) > num_structures:
                structures.pop(0)

        os.makedirs(output_folder, exist_ok=True)
        for idx, structure in enumerate(structures):
            subdir = os.path.join(output_folder, f"coord_{idx}")
            os.makedirs(subdir, exist_ok=True)
            output_file = os.path.join(subdir, 'coord.sys')
            with open(output_file, 'w') as coord_file:
                coord_file.write("&COORD\n")
                coord_file.writelines(structure)
                coord_file.write("&END COORD\n")

        print(f"Successfully saved last {num_structures} structures to: {output_folder}")

    except Exception as e:
        print(f"Failed to process file: {file_path} with error: {e}")

def main():
    """
    Main function to parse arguments and invoke the structure extraction process.
    """
    parser = argparse.ArgumentParser(description="Process the last structures from an XYZ file and save them.")
    parser.add_argument("--input", "-i", required=True, help="Path to the input XYZ file.")
    parser.add_argument("--output", "-o", required=True, help="Path to the output folder where the results will be saved.")
    parser.add_argument("--num", "-n", type=int, default=10, help="Number of most recent structures to save (default: 10).")

    args = parser.parse_args()

    if not os.path.isfile(args.input):
        print(f"Error: File not found - {args.input}")
        return 0

    save_last_structures(args.input, args.output, args.num)

if __name__ == "__main__":
    main()

