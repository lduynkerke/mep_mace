import argparse
import random
import time
import numpy as np
from ase import units
from ase.io import read, write, Trajectory
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary, ZeroRotation
from mace.calculators import MACECalculator

def run_md_for_model(dyn, atom_conf, T0, calc, Niter, traj_file=None):
    """
    Runs molecular dynamics simulation for a single model.
    Optionally writes the trajectory to a specified file.

    :param dyn: ASE dynamics object (e.g., NPT, NVT, etc.).
    :type dyn: ase.md.Dynamics
    :param atom_conf: Initial atomic configuration.
    :type atom_conf: ase.Atoms
    :param T0: Initial temperature in Kelvin.
    :type T0: float
    :param calc: The calculator object to compute forces and energies during the simulation.
    :type calc: ase.Calculator
    :param Niter: The number of MD steps to simulate.
    :type Niter: int
    :param traj_file: Path to the file where the trajectory is stored (optional).
    :type traj_file: str, optional
    :return: A list of potential energies from the simulation.
    :rtype: list[float]
    """
    def write_frame():
        if traj_file is not None:
            dyn.atoms.write("all.xyz", append=True)
            dyn.atoms.write(traj_file, append=True)
        time_fs.append(dyn.get_time() / units.fs)
        temperature.append(dyn.atoms.get_temperature())
        energies.append(dyn.atoms.get_potential_energy() / len(dyn.atoms))

    random.seed(701)
    atom_conf.set_calculator(calc)
    MaxwellBoltzmannDistribution(atom_conf, temperature_K=T0)
    Stationary(atom_conf)
    ZeroRotation(atom_conf)

    time_fs, temperature, energies = [], [], []
    dyn.attach(write_frame)
    dyn.run(Niter)

    return energies


def run_simulation(args, init_conf, nose):
    """
    Run the molecular dynamics simulation with multiple models.
    
    This function calculates and evaluates the standard deviation of energies
    across models until a convergence criterion is met or the maximum number
    of iterations is reached.

    :param args: Parsed command-line arguments containing configuration.
    :param init_conf: Initial atomic configuration for the simulation.
    :param nose: Nose-Hoover thermostat for molecular dynamics.
    """
    all_energies = [[] for _ in args.models]  
    first_model_written = False
    std_dev, iters = 0.0, 0

    while std_dev < args.std_threshold and iters < args.nmax:
        with open(args.traj_name, 'w') as file0:
            pass

        for i, model in enumerate(args.models):
            print(f"Running model {i + 1}/{len(args.models)}")
            mace_calc = MACECalculator(model_paths=model, device='cuda')
            traj_file = args.traj_name if i ==0 else None
            energies = run_md_for_model(
                    dyn=nose, 
                    atom_conf=init_conf, 
                    T0=args.T0, 
                    calc=mace_calc, 
                    Niter=args.niter,
                    traj_file=args.traj_name
            )
            all_energies[i].extend(energies)

        if len(args.models) > 1:
            energy_matrix = np.array(all_energies, dtype=object)
            std_dev = np.mean([np.std(step) for step in zip(*energy_matrix) if step])
            print(f"Current standard deviation: {std_dev}")

        iters += args.niter
    
    print(all_energies)
    print("Simulation completed.")


def main():
    """
    Runs a molecular dynamics simulation with multiple models and a Nose-Hoover thermostat.
    """
    parser = argparse.ArgumentParser(description="Run molecular dynamics simulation.")
    parser.add_argument("--models", required=True, nargs='*', help="Path to the model file.")
    parser.add_argument("--structure", required=True, help="Path to the structure file.")
    parser.add_argument("--temp", type=float, default=353, help="Eventual temperature in Kelvin.")
    parser.add_argument("--T0", type=float, default=300, help="Initial temperature in Kelvin.")
    parser.add_argument("--niter", type=int, default=10, help="Number of MD steps.")
    parser.add_argument("--nmax", type=int, default=10000, help="Maximum number of MD steps")
    parser.add_argument("--traj_name", default="trajectory.xyz", help="Trajectory output file name.")
    parser.add_argument("--logfile", default="md_nose.log", help="Log file for md.")
    parser.add_argument("--std_threshold", type=float, default=0.1, help="Standard deviation threshold.")
    args = parser.parse_args()

    # Nose-Hoover thermostat
    init_conf = read(args.structure)
    nose = NPT(
        atoms=init_conf,
        timestep=0.5*units.fs,
        temperature_K=args.temp,
        ttime=15*units.fs,
        externalstress=None,  # No stress for NVT
        logfile=args.logfile
    )
    
    # Run MD
    run_simulation(args, init_conf, nose)

if __name__ == "__main__":
    main()

