import numpy as np
import matplotlib.pyplot as plt
import re

def load_energies(filename, tag='TotEnergy'):
    energies = []
    with open(filename, 'r') as file:
        for line in file:
            # Match lines with TotEnergy values
            if tag in line:
                match = re.search(tag+'\s*=\s*([-\d.]+)', line)
                if match:
                    energy = float(match.group(1))
                    energies.append(energy)
    return np.array(energies)

def main():
    mace_energies = load_energies('traj/nose_773K.xyz', tag='energy')[5000:]/506
    pbe_energies = load_energies('../Forces_and_traj_353_1.xyz', tag='TotEnergy')[5000:]/506
    pbe_mean = np.mean(pbe_energies)
    pbe_variance = np.var(pbe_energies)
    mace_mean = np.mean(mace_energies)
    mace_variance = np.var(mace_energies)

    # Print statistical information
    print("PBE Energies:")
    print(f"Mean: {pbe_mean}, Variance: {pbe_variance}")
    print("MACE Energies:")
    print(f"Mean: {mace_mean}, Variance: {mace_variance}")

    # Plot normalized energies from pbe and mace
    plt.figure(figsize=(10, 5))
    plt.plot(pbe_energies, label='PBE')
    plt.plot(mace_energies, label='MACE', linestyle='--')
    plt.title("Normalized Energies")
    plt.xlabel("Time step")
    plt.ylabel("Energy (eV/atom)")
    plt.legend()
    plt.savefig("energies.png")
    plt.show()

    # Additional plots and Figures
    # Histogram of pbe energies
    plt.figure()
    plt.hist(pbe_energies, bins=250, range=(-10.93, -10.87), alpha=0.7, color='blue', edgecolor='black', label='PBE')
    plt.hist(mace_energies, bins=250, range=(-10.93, -10.82), alpha=0.7, color='orange', edgecolor='black', label='MACE')
    plt.title("Histogram of Energies")
    plt.xlabel("Energy (eV/atom)")
    plt.ylabel("Frequency")
    plt.legend()
    plt.savefig("energy_hist.png")
    plt.show()

    # Scatter plot for comparison
    plt.figure()
    plt.scatter(range(len(pbe_energies)), pbe_energies, alpha=0.6, label='PBE')
    plt.scatter(range(len(mace_energies)), mace_energies, alpha=0.6, label='MACE')
    plt.title("Scatter Plot: Energies")
    plt.xlabel("Data Point")
    plt.ylabel("Energy (eV/atom)")
    plt.legend()
    plt.show()

    # Autocorrelation plot for pbe energies
    plt.figure()
    pbe_autocorr = np.correlate(pbe_energies - pbe_mean, pbe_energies - pbe_mean, mode='full')
    pbe_autocorr = pbe_autocorr[pbe_autocorr.size // 2:]
    plt.plot(pbe_autocorr / pbe_autocorr[0], label="PBE")

    # mace_autocorr = np.correlate(mace_energies - mace_mean, mace_energies - mace_mean, mode='full')
    # mace_autocorr = mace_autocorr[mace_autocorr.size // 2:]
    # plt.plot(mace_autocorr / mace_autocorr[0], label="MACE")

    plt.title("Autocorrelation of Energies")
    plt.xlabel("Lag")
    plt.ylabel("Autocorrelation")
    plt.savefig("corr_pbe.png")
    plt.legend()
    plt.show()

    # Box plot for both datasets
    plt.figure()
    plt.boxplot([pbe_energies, mace_energies], labels=['PBE Energies', 'mace Energies'])
    plt.title("Box Plot of Normalized Energies")
    plt.ylabel("Energy (eV/atom)")
    plt.savefig("energy_box.png")
    plt.show()

if __name__ == "__main__":
    main()