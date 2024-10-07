import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

def load_data(base_path, strain_surface, sim_type, temp_folder):
    """
    Loads CSV data for a specific strain surface, simulation type, and temperature folder.

    This function constructs the path to a CSV file based on the provided parameters and
    loads the data into a pandas DataFrame. If the file does not exist, it returns an
    empty DataFrame with columns ['Extracted Energy', 'Predicted Energy'].

    :param base_path: The base directory where the CSV files are stored.
    :type base_path: str
    :param strain_surface: The name of the strain surface folder.
    :type strain_surface: str
    :param sim_type: The name of the simulation type folder.
    :type sim_type: str
    :param temp_folder: The name of the temperature folder.
    :type temp_folder: str
    :return: A pandas DataFrame containing the loaded data or an empty DataFrame if the file doesn't exist.
    :rtype: pd.DataFrame
    """
    csv_path = os.path.join(base_path, strain_surface, sim_type, temp_folder, 'mace0.csv')
    if os.path.exists(csv_path):
        try:
            return pd.read_csv(csv_path)
        except pd.errors.EmptyDataError:
            print("Error reading", csv_path)
            return pd.DataFrame(columns=['Extracted Energy', 'Predicted Energy'])
    else:
        return pd.DataFrame(columns=['Extracted Energy', 'Predicted Energy'])


def plot_simulation_data(base_path, sim_type, title, strain_surfaces, temperature_folders):
    """
    Generates scatter plots comparing 'Extracted Energy' and 'Predicted Energy' for different simulations.

    This function loads the relevant data from CSV files, creates scatter plots for each combination
    of strain surface and temperature folder, and returns a matplotlib figure.

    :param base_path: The base directory where the CSV files are stored.
    :type base_path: str
    :param sim_type: The simulation type folder to analyze.
    :type sim_type: str
    :param title: The title for the figure.
    :type title: str
    :param strain_surfaces: The names of the strain surface folders to analyze.
    :type strain_surfaces: list
    :param temperature_folders: The names of the temperature folders to analyze.
    :type temperature_folders: list
    :return: A matplotlib figure containing the scatter plots.
    :rtype: matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(title)

    for i, strain_surface in enumerate(strain_surfaces):
        for j, temp_folder in enumerate(temperature_folders):
            data = load_data(base_path, strain_surface, sim_type, temp_folder)
            ax = axes[i, j]
            if not data.empty:
                length = data.shape[0]
                ax.scatter(data['Extracted Energy']/length, data['Predicted Energy']/length)
                ax.set_title(f'{strain_surface} - {temp_folder}')
                if i == 2:
                    ax.set_xlabel('DFT Energy / atom')
                if j == 0:
                    ax.set_ylabel('MACE Energy / atom')
                ax.set_xlim(-5528, -5524)
                ax.set_ylim(-3570, -3500)
                ax.set_xticks([-5528, -5527, -5526, -5525, -5524])
                ax.set_yticks([-3570, -3560, -3550, -3540, -3530, -3520, -3510, -3500])
            else:
                ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
                ax.set_title(f'{strain_surface} - {temp_folder}')
            ax.grid(True)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))  # Adjust layout to make space for the title
    return fig


def plot_simulation_data_one(base_path, sim_types, sim_type_legends, title, strain_surfaces, strain_surface_names,
                         temperature_folders, temp_folder_names):
    """
    Generates scatter plots comparing 'Extracted Energy' and 'Predicted Energy' for different simulations.

    This function loads the relevant data from CSV files, creates scatter plots for each combination
    of strain surface and temperature folder, and returns a matplotlib figure.

    :param base_path: The base directory where the CSV files are stored.
    :type base_path: str
    :param sim_types: A list of simulation type folders to analyze.
    :type sim_types: list
    :param sim_type_legends: A list of legends corresponding to each sim_type.
    :type sim_type_legends: list
    :param title: The title for the figure.
    :type title: str
    :param strain_surfaces: The names of the strain surface folders to analyze.
    :type strain_surfaces: list
    :param strain_surface_names: A list of custom names for the strain surfaces.
    :type strain_surface_names: list
    :param temperature_folders: The names of the temperature folders to analyze.
    :type temperature_folders: list
    :param temp_folder_names: A list of custom names for the temperature folders.
    :type temp_folder_names: list
    :return: A matplotlib figure containing the scatter plots.
    :rtype: matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(3, 3, figsize=(15, 15))
    fig.suptitle(title)

    # Use a color map for different sim_types
    colors = list(mcolors.TABLEAU_COLORS.keys())  # Choose a palette of colors

    for sim_idx, sim_type in enumerate(sim_types):
        color = colors[sim_idx % len(colors)]  # Cycle through colors if more sim_types than colors

        for i, strain_surface in enumerate(strain_surfaces):
            for j, temp_folder in enumerate(temperature_folders):
                data = load_data(base_path, strain_surface, sim_type, temp_folder)
                ax = axes[i, j]

                if not data.empty:
                    ax.scatter(data['Extracted Energy']/505, data['Predicted Energy']/505, color=color,
                               label=sim_type_legends[sim_idx])

                    # Set custom titles using strain_surface_names and temp_folder_names
                    strain_surface_name = strain_surface_names[i] if i < len(strain_surface_names) else strain_surface
                    temp_folder_name = temp_folder_names[j] if j < len(temp_folder_names) else temp_folder
                    ax.set_title(f'{strain_surface_name} - {temp_folder_name}')

                    # Only add labels on the outer axes
                    if i == 2:
                        ax.set_xlabel('DFT Energy / atom')
                    if j == 0:
                        ax.set_ylabel('MACE Energy / atom')

                    ax.set_xlim(-10.947, -10.938)
                    ax.set_ylim(-7.08, -6.94)
                    ax.set_xticks([-10.946, -10.944, -10.942, -10.94, -10.938])
                    ax.set_yticks([-7.08, -7.06, -7.04, -7.02, -7.0, -6.98, -6.96, -6.94])
                else:
                    ax.text(0.5, 0.5, 'No Data', horizontalalignment='center', verticalalignment='center')
                    strain_surface_name = strain_surface_names[i] if i < len(strain_surface_names) else strain_surface
                    temp_folder_name = temp_folder_names[j] if j < len(temp_folder_names) else temp_folder
                    ax.set_title(f'{strain_surface_name} - {temp_folder_name}')

                ax.grid(False)

    # Add legends to the axes
    for ax in axes.flatten():
        ax.legend(loc=2)

    plt.tight_layout(rect=(0.0, 0.03, 1.0, 0.95))  # Adjust layout to make space for the title
    return fig


def main():
    """
    The main function that generates and saves scatter plot figures for different simulation types.

    This function creates scatter plots for two simulation types ('Adsorption Complex Simulations'
    and 'Active ZrH Site Simulations') and saves them as PNG files. The plots compare 'Extracted Energy'
    and 'Predicted Energy' for each combination of strain surface and temperature folder.

    :return: None
    """
    base_dir = r"../Results"
    simulation_types = ["Adsorption_Complex_Simulations", "Active_ZrH_site_simulations"]
    simulation_types_legend = ["AC_Sim", "ZrH_Sim"]
    strain_surfaces = ["min_strain_surface", "av_strain_surface", "max_strain_surface"]
    strain_surfaces_names = ["s_min", "s_av", 's_max']
    temperature_folders = ["aiMD_353K", "aiMD_773K", "Velocity_softening_dynamics"]
    temperature_folders_names = ["353K", "773K", "VSD"]

    # Create and save figures
    fig0 = plot_simulation_data_one(base_path=base_dir,
                                    sim_types=simulation_types,
                                    sim_type_legends=simulation_types_legend,
                                    title="Results MACE-MP-0",
                                    strain_surfaces=strain_surfaces,
                                    strain_surface_names=strain_surfaces_names,
                                    temperature_folders=temperature_folders,
                                    temp_folder_names=temperature_folders_names
                                    )

    # fig1 = plot_simulation_data(base_path=base_dir,
    #                             sim_type=simulation_types[0],
    #                             title="Adsorption Complex Simulations",
    #                             strain_surfaces=strain_surfaces,
    #                             temperature_folders=temperature_folders
    #                             )
    #
    # fig2 = plot_simulation_data(base_path=base_dir,
    #                             sim_type=simulation_types[1],
    #                             title="Active ZrH Site Simulations",
    #                             strain_surfaces=strain_surfaces,
    #                             temperature_folders=temperature_folders
    #                             )
    fig0.savefig(os.path.join(base_dir, 'MACE-MP-0.png'))
    # fig1.savefig(os.path.join(base_dir, 'Adsorption_Complex_Simulations_MACE0_Autoscale.png'))
    # fig2.savefig(os.path.join(base_dir, 'Active_ZrH_Site_Simulations_MACE0_Autoscale.png'))

    # Show the figures
    plt.show()


if __name__ == "__main__":
    main()
