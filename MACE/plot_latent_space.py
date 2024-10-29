import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from mace import data, modules, tools
import torch
import ase

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = torch.load("../av_adsorption_353K.model")
    Rcut = 3.0  # cutoff radius
    z_table = tools.AtomicNumberTable([1, 6, 8, 14, 40])
    all_node_features, all_atomic_properties, all_edge_features, all_edge_attributes = [], [], [], []

    ### 0 DATA PREP
    molecules = ase.io.read('../merged.xyz', index=':')

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
        edge_features = model.radial_embedding(lengths, batch["node_attrs"], batch["edge_index"], atomic_numbers).detach().cpu().numpy()
        edge_attributes = model.spherical_harmonics(vectors).detach().cpu().numpy()
        # node_features = initial_node_features.detach().cpu().numpy()
        # edge_features = initial_node_features.detach().cpu().numpy()
        atom_properties = batch['node_attrs'].detach().cpu().numpy()

        # Append the features and properties to the lists
        all_node_features.append(node_features)
        all_atomic_properties.append(atom_properties)
        all_edge_features.append(edge_features)
        all_edge_attributes.append(edge_attributes)

    # Stack the data for PCA and t-SNE
    all_node_features = np.vstack(all_node_features)
    all_atomic_properties = np.vstack(all_atomic_properties)
    all_edge_features = np.vstack(all_edge_features)
    all_edge_attributes = np.vstack(all_edge_attributes)

    # Perform PCA and t-SNE on the stacked features
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # PCA plot
    pca = PCA(n_components=2)
    node_features_2d_pca = pca.fit_transform(all_node_features)
    edge_features_2d_pca = pca.fit_transform(all_edge_features)
    edge_attributes_2d_pca = pca.fit_transform(all_edge_attributes)
    axs[0].scatter(edge_attributes_2d_pca[:, 0], edge_attributes_2d_pca[:, 1], c='C2', label='attributes')
    axs[0].scatter(edge_features_2d_pca[:, 0], edge_features_2d_pca[:, 1], c='C0', label='edges')
    axs[0].scatter(node_features_2d_pca[:, 0], node_features_2d_pca[:, 1], c='C1', label='nodes')
    axs[0].set_title("PCA of Node Latent Space")
    axs[0].set_xlabel("PC 1")
    axs[0].set_ylabel("PC 2")
    axs[0].get_legend()
    #plt.colorbar(scatter_pca, ax=axs[0], label="Atomic Property")

    # t-SNE plot
    tsne = TSNE(n_components=2, perplexity=30, n_iter=300)
    node_features_2d_tsne = tsne.fit_transform(all_node_features)
    edge_features_2d_tsne = tsne.fit_transform(all_edge_features)
    axs[1].scatter(edge_features_2d_tsne[:, 0], edge_features_2d_tsne[:, 1], c='C0', label='edges')
    axs[1].scatter(node_features_2d_tsne[:, 0], node_features_2d_tsne[:, 1], c='C1', label='nodes')
    axs[1].set_title("t-SNE of Node Latent Space")
    axs[1].set_xlabel("t-SNE 1")
    axs[1].set_ylabel("t-SNE 2")
    #plt.colorbar(scatter_tsne, ax=axs[1], label="Atomic Property")

    plt.tight_layout()
    plt.savefig("latent_space_tsne.png")
    plt.show()
    #

    # Perform PCA on the stacked features
    fig, axs = plt.subplots(1, 2, figsize=(14, 6))

    # 2D PCA plot for ax[0]
    pca_2d = PCA(n_components=2)
    node_features_2d_pca = pca_2d.fit_transform(all_node_features)
    edge_features_2d_pca = pca_2d.fit_transform(all_edge_features)
    edge_attributes_2d_pca = pca_2d.fit_transform(all_edge_attributes)

    axs[0].scatter(edge_attributes_2d_pca[:, 0], edge_attributes_2d_pca[:, 1], c='C2', label='Attributes')
    axs[0].scatter(edge_features_2d_pca[:, 0], edge_features_2d_pca[:, 1], c='C0', label='Edges')
    axs[0].scatter(node_features_2d_pca[:, 0], node_features_2d_pca[:, 1], c='C1', label='Nodes')
    axs[0].set_title("PCA of Node Latent Space (2D)")
    axs[0].set_xlabel("PC 1")
    axs[0].set_ylabel("PC 2")
    axs[0].legend()  # Add legend for the 2D plot

    # 3D PCA plot for ax[1]
    pca_3d = PCA(n_components=3)
    node_features_3d_pca = pca_3d.fit_transform(all_node_features)
    edge_features_3d_pca = pca_3d.fit_transform(all_edge_features)
    edge_attributes_3d_pca = pca_3d.fit_transform(all_edge_attributes)

    # Creating a 3D subplot for PCA
    ax1 = fig.add_subplot(122, projection='3d')
    ax1.scatter(edge_attributes_3d_pca[:, 0], edge_attributes_3d_pca[:, 1], edge_attributes_3d_pca[:, 2], c='C2',
                label='Attributes')
    ax1.scatter(edge_features_3d_pca[:, 0], edge_features_3d_pca[:, 1], edge_features_3d_pca[:, 2], c='C0',
                label='Edges')
    ax1.scatter(node_features_3d_pca[:, 0], node_features_3d_pca[:, 1], node_features_3d_pca[:, 2], c='C1',
                label='Nodes')
    ax1.set_title("PCA of Node Latent Space (3D)")
    ax1.set_xlabel("PC 1")
    ax1.set_ylabel("PC 2")
    ax1.set_zlabel("PC 3")
    ax1.legend()  # Add legend for the 3D plot

    plt.tight_layout()
    plt.savefig("latent_space_pca.png")
    plt.show()

if __name__ == '__main__':
    main()