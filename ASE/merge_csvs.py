import os
import pandas as pd
import argparse


def rename_and_merge_csvs(base_directory):
    """
    Renames CSV files in a directory and merges them into a single file.

    This function traverses a given directory, renames all CSV files that match the pattern
    'Forces_and_traj_X.csv' to 'mace0_X.csv', and then merges these files into a single CSV
    named 'mace0.csv' in the same directory.

    :param base_directory: The base directory where CSV files are located.
    :type base_directory: str
    :return: None
    """
    for root, dirs, files in os.walk(base_directory):
        csv_files = sorted([f for f in files if f.startswith("Forces_and_traj_") and f.endswith(".csv")])

        if not csv_files:
            continue

        dataframes = []

        # Process and rename the files
        for i, csv_file in enumerate(csv_files):
            old_file_path = os.path.join(root, csv_file)
            new_file_name = f"mace0_{i + 1}.csv"
            new_file_path = os.path.join(root, new_file_name)

            os.rename(old_file_path, new_file_path)
            print(f"Renamed: {old_file_path} -> {new_file_name}")

            df = pd.read_csv(new_file_path)
            dataframes.append(df)

        merged_df = pd.concat(dataframes, ignore_index=True)

        merged_csv_path = os.path.join(root, "mace0.csv")
        merged_df.to_csv(merged_csv_path, index=False)
        print(f"Merged CSV saved as: {merged_csv_path}")


def main():
    """
    The main function to parse command-line arguments and execute the CSV renaming and merging process.

    This function specifies the base directory where the CSV files are located and triggers the
    renaming and merging process.

    :return: None
    """
    parser = argparse.ArgumentParser(description="Rename and merge CSV files.")
    parser.add_argument('base_directory', help="The base directory where CSV files are located.")

    args = parser.parse_args()
    rename_and_merge_csvs(args.base_directory)


if __name__ == "__main__":
    main()