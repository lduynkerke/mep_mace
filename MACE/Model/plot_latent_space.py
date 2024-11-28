import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mace import data, modules, tools
import torch
import ase

def get_features(file, model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load(model)
    Rcut = 3.0  # cutoff radius
    z_table = tools.AtomicNumberTable([1, 6, 8, 14, 40])
    all_node_features, all_atomic_properties, all_edge_features, all_edge_attributes = [], [], [], []

    molecules = ase.io.read(file, index=':')
    for single_molecule in molecules:
        config = data.Configuration(
            atomic_numbers=single_molecule.numbers,
            positions=single_molecule.positions
        )
        batch = data.AtomicData.from_config(config, z_table=z_table, cutoff=Rcut)
        vectors, lengths = modules.utils.get_edge_vectors_and_lengths(
            positions=batch["positions"],
            edge_index=batch["edge_index"],
            shifts=batch["shifts"],
        )

        # Process molecule data as a batch
        batch.node_attrs = batch.node_attrs.to(torch.float64).to(device)
        batch["positions"] = batch["positions"].to(torch.float64).to(device)
        batch.shifts = batch.shifts.to(torch.float64).to(device)
        batch.edge_index = batch.edge_index.to(device)
        batch.unit_shifts = batch.unit_shifts.to(device)
        batch.cell = batch.cell.to(device)
        batch.weight = batch.weight.to(device)
        batch.energy_weight = batch.energy_weight.to(device)
        batch.forces_weight = batch.forces_weight.to(device)
        batch.stress_weight = batch.stress_weight.to(device)
        batch.virials_weight = batch.virials_weight.to(device)
        lengths = lengths.to(device)
        vectors = vectors.to(device)

        # Obtain atomic numbers from node attributes
        atomic_numbers = torch.argmax(batch.node_attrs, dim=1)
        atomic_numbers = torch.tensor([1, 6, 8, 14, 40], device=device)[atomic_numbers]

        # Calculate embeddings
        node_features = model.node_embedding(batch.node_attrs).detach().cpu().numpy()
        edge_features = model.radial_embedding(lengths, batch["node_attrs"], batch["edge_index"],
                                               atomic_numbers).detach().cpu().numpy()
        edge_attributes = model.spherical_harmonics(vectors).detach().cpu().numpy()
        atom_properties = batch['node_attrs'].detach().cpu().numpy()

        # Append the features and properties to the lists and convert to numpy arrays
        all_node_features.append(node_features)
        all_atomic_properties.append(atom_properties)
        all_edge_features.append(edge_features)
        all_edge_attributes.append(edge_attributes)

        all_node_features = np.vstack(all_node_features)
        all_atomic_properties = np.vstack(all_atomic_properties)
        all_edge_features = np.vstack(all_edge_features)
        all_edge_attributes = np.vstack(all_edge_attributes)

        return [all_node_features, all_atomic_properties, all_edge_features, all_edge_attributes]

def pca_2d_3d(model_features, filename='latent_space_pca_2D_3D.png'):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # 2D
    pca_2d = PCA(n_components=2)
    node_features_2d_pca = pca_2d.fit_transform(model_features[0])
    edge_features_2d_pca = pca_2d.fit_transform(model_features[2])
    # edge_attributes_2d_pca = pca_2d.fit_transform(model_features[3])
    # axs[0].scatter(edge_attributes_2d_pca[:, 0], edge_attributes_2d_pca[:, 1], c='C2', label='Attributes')
    axs[0].scatter(edge_features_2d_pca[:, 0], edge_features_2d_pca[:, 1], c='C0', label='Edges')
    axs[0].scatter(node_features_2d_pca[:, 0], node_features_2d_pca[:, 1], c='C1', label='Nodes')
    axs[0].set_title("PCA of Node Latent Space (2D)")
    axs[0].set_xlabel("PC 1")
    axs[0].set_ylabel("PC 2")
    axs[0].legend()

    # 3D
    pca_3d = PCA(n_components=3)
    ax1 = fig.add_subplot(122, projection='3d')
    node_features_3d_pca = pca_3d.fit_transform(model_features[0])
    edge_features_3d_pca = pca_3d.fit_transform(model_features[2])
    # edge_attributes_3d_pca = pca_3d.fit_transform(model_features[3])
    # ax1.scatter(edge_attributes_3d_pca[:, 0], edge_attributes_3d_pca[:, 1], edge_attributes_3d_pca[:, 2], c='C2',
    #            label='Attributes')
    ax1.scatter(edge_features_3d_pca[:, 0], edge_features_3d_pca[:, 1], edge_features_3d_pca[:, 2], c='C0',
                label='Edges')
    ax1.scatter(node_features_3d_pca[:, 0], node_features_3d_pca[:, 1], node_features_3d_pca[:, 2], c='C1',
                label='Nodes')
    ax1.set_title("PCA of Node Latent Space (3D)")
    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    ax1.set_zlabel("PC 3")
    ax1.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def pca_tsne(model_features, filename='latent_space_pca_tsne.png'):
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # PCA
    pca = PCA(n_components=2)
    node_features_2d_pca = pca.fit_transform(model_features[0])
    edge_features_2d_pca = pca.fit_transform(model_features[2])
    # edge_attributes_2d_pca = pca_2d.fit_transform(model_features[3])
    # axs[0].scatter(edge_attributes_2d_pca[:, 0], edge_attributes_2d_pca[:, 1], c='C2', label='attributes')
    axs[0].scatter(edge_features_2d_pca[:, 0], edge_features_2d_pca[:, 1], c='C0', label='edges')
    axs[0].scatter(node_features_2d_pca[:, 0], node_features_2d_pca[:, 1], c='C1', label='nodes')
    axs[0].set_title("PCA of Node Latent Space")
    axs[0].set_xlabel("PC 1")
    axs[0].set_ylabel("PC 2")
    axs[0].get_legend()
    #plt.colorbar(scatter_pca, ax=axs[0], label="Atomic Property")

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    node_features_2d_tsne = tsne.fit_transform(model_features[0])
    edge_features_2d_tsne = tsne.fit_transform(model_features[2])
    axs[1].scatter(edge_features_2d_tsne[:, 0], edge_features_2d_tsne[:, 1], c='C0', label='edges')
    axs[1].scatter(node_features_2d_tsne[:, 0], node_features_2d_tsne[:, 1], c='C1', label='nodes')
    axs[1].set_title("t-SNE of Node Latent Space")
    axs[1].set_xlabel("t-SNE 1")
    axs[1].set_ylabel("t-SNE 2")
    #plt.colorbar(scatter_tsne, ax=axs[1], label="Atomic Property")

    plt.tight_layout()
    plt.savefig("latent_space_tsne.png")
    plt.show()

def compare_pca(features1, features2, color1, color2, filename):
    pca_transformed_points_1, colors_1 = [], []
    pca_transformed_points_2, colors_2 = [], []
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # PCA1
    pca_2d_1 = PCA(n_components=2)
    # for i in range(features1[2].shape[0]):
    #     edge_feature = features1[2][i, 0].reshape(-1, 1)
    #     pca_result = pca_2d_1.fit_transform(edge_feature)
    #     pca_transformed_points_1.append(pca_result)
    #     colors_1.extend( np.repeat(color1[i], pca_transformed_points_1[0].shape[0]))
    # edge_features_2d_pca_1 = np.vstack(pca_transformed_points_1)
    edge_features_2d_pca_1 = pca_2d_1.fit_transform(features1[2])
    node_features_2d_pca_1 = pca_2d_1.fit_transform(features1[0])

    sc1 = axs[0].scatter(node_features_2d_pca_1[:, 0], node_features_2d_pca_1[:, 1], c='C1', label='Nodes')
    sc1 = axs[0].scatter(edge_features_2d_pca_1[:, 0], edge_features_2d_pca_1[:, 1], c='C0', label='Nodes')
    # sc2 = axs[0].scatter(
    #     edge_features_2d_pca_1[:, 0], edge_features_2d_pca_1[:, 1],
    #     c=color1, cmap='coolwarm', label='Edges', marker='x'
    # )
    axs[0].set_title("PCA of Node and Edge Features (Max)")
    axs[0].set_xlabel("PC 1")
    axs[0].set_ylabel("PC 2")
    axs[0].legend()
    #cbar = plt.colorbar(sc1, ax=axs[0])
    #cbar.set_label("Error")

    # PCA 2
    pca_2d_2 = PCA(n_components=2)
    # for i in range(features2[2].shape[0]):
    #     edge_feature = features2[2][i, 0].reshape(-1, 1)
    #     pca_result = pca_2d_2.fit_transform(edge_feature)
    #     pca_transformed_points_2.append(pca_result)
    #     colors_2.extend( np.repeat(color2[i], pca_transformed_points_2[0].shape[0]))
    # edge_features_2d_pca_2 = np.vstack(pca_transformed_points_2)
    node_features_2d_pca_2 = pca_2d_2.fit_transform(features2[0])
    edge_features_2d_pca_2 = pca_2d_2.fit_transform(features1[2])

    sc3 = axs[1].scatter(node_features_2d_pca_2[:, 0], node_features_2d_pca_2[:, 1], c='C1', label='Nodes')
    sc3 = axs[1].scatter(edge_features_2d_pca_2[:, 0], edge_features_2d_pca_2[:, 1], c='C0', label='Nodes')
    # sc4 = axs[1].scatter(
    #     edge_features_2d_pca_2[:, 0], edge_features_2d_pca_2[:, 1],
    #     c=color2, cmap='coolwarm', label='Edges', marker='x'
    # )
    axs[1].set_title("PCA of Node and Edge Features (Min)")
    axs[1].set_xlabel("PC 1")
    axs[1].set_ylabel("PC 2")
    axs[1].legend()
    # cbar = plt.colorbar(sc3, ax=axs[1])
    # cbar.set_label("Error")

    plt.tight_layout()
    plt.savefig(filename)
    plt.show()

def main():
    features_max = get_features("../../merged.xyz", "../av_adsorption_353K.model")
    features_min = get_features("../../merged_correct.xyz", "../av_adsorption_353K.model")
    pca_2d_3d(features_max)

    energies_max_mace = np.load(
        '../../Results/max_strain_surface/Adsorption_Complex_Simulations/Velocity_softening_dynamics/mace1_forces.npz')['energies']
    energies_min_mace = np.load(
        '../../Results/min_strain_surface/Adsorption_Complex_Simulations/aiMD_773K/mace1_forces.npz')['energies']
    energies_max_pbe = np.load(
        '../../Results/max_strain_surface/Adsorption_Complex_Simulations/Velocity_softening_dynamics/PBE_forces.npz')['energies']
    energies_min_pbe = np.load(
        '../../Results/min_strain_surface/Adsorption_Complex_Simulations/aiMD_773K/PBE_forces.npz')['energies']
    energy_err_max = np.abs(energies_max_mace - energies_max_pbe)
    energy_err_min = np.abs(energies_min_mace - energies_min_pbe)
    idx = np.arange(0, energies_max_mace.shape[0], energies_max_mace.shape[0])

    # pca_tsne = (features_max)
    compare_pca(features_max, features_min, energy_err_max, energy_err_min, "pca_by_error.png")
    # compare_pca(features_min, features_max, idx, idx, "pca_by_index.png")

if __name__ == '__main__':
    main()