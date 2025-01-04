import numpy as np
import pylab as pl
import time
import random
from mace.calculators import MACECalculator
from matscipy.neighbours import neighbour_list
from ASE.aseMolec import extAtoms as ea
from ase import Atoms, units
from ase.io import read, write
from ase.neighborlist import natural_cutoffs
from ase.md.npt import NPT
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution


def run_md(dyn, atom_conf, T0, calc, Niter, traj_name, img_name, max_force_threshold=None):
    """
    Executes a molecular dynamics (MD) simulation and plots the results.

    This function initializes the MD simulation by setting up the atom configuration
    with a given calculator and assigning an initial temperature using the Maxwell-Boltzmann
    distribution. It then runs the simulation until either the maximum number of iterations
    is reached or the maximum atomic force exceeds the specified threshold.

    :param dyn: The molecular dynamics driver to execute the simulation.
    :type dyn: ase.md.Verlet or other ASE MD driver
    :param atom_conf: The atomic configuration to be simulated.
    :type atom_conf: ase.Atoms
    :param T0: The initial temperature of the system in Kelvin.
    :type T0: float
    :param calc: The calculator object to compute forces and energies during the simulation.
    :type calc: ase.Calculator
    :param Niter: The number of MD steps to simulate.
    :type Niter: int
    :param traj_name: Filename for saving the trajectory of the simulation.
    :type traj_name: str
    :param img_name: Filename for saving the energy and temperature plot.
    :type img_name: str
    :param max_force_threshold: Maximum allowed force magnitude (eV/Å). If exceeded, simulation stops.
    :type max_force_threshold: float or None
    :return: dict containing simulation statistics and reason for termination
    """

    def write_frame():
        dyn.atoms.write(traj_name, append=True)
        time_fs.append(dyn.get_time() / units.fs)
        temperature.append(dyn.atoms.get_temperature())
        energies.append(dyn.atoms.get_potential_energy() / len(dyn.atoms))

        # Calculate maximum force if threshold is specified
        if max_force_threshold is not None:
            forces = dyn.atoms.get_forces()
            max_force = np.max(np.sqrt(np.sum(forces ** 2, axis=1)))
            max_forces.append(max_force)
            return max_force
        return None

    # Initialize
    random.seed(701)
    atom_conf.set_calculator(calc)
    MaxwellBoltzmannDistribution(atom_conf, temperature_K=T0)
    Stationary(atom_conf)
    ZeroRotation(atom_conf)

    # Run MD with force monitoring
    time_fs, temperature, energies, max_forces = [], [], [], []
    dyn.attach(write_frame)
    t0 = time.time()

    termination_reason = "Completed all iterations"
    try:
        for i in range(Niter):
            dyn.run(1)  # Run one step at a time
            if i % 100 == 0:
                print(f"Iteration {i}")
            if max_force_threshold is not None and max_forces[-1] > max_force_threshold:
                termination_reason = f"Force threshold {max_force_threshold:.2f} eV/Å exceeded"
                break
    except Exception as e:
        termination_reason = f"Error occurred: {str(e)}"

    t1 = time.time()
    runtime_minutes = (t1 - t0) / 60
    print(f"MD {termination_reason} in {runtime_minutes:.2f} minutes!")

    # Plot energy, temperature, and forces if monitored
    n_plots = 3 if max_force_threshold is not None else 2
    fig, ax = pl.subplots(n_plots, 1, figsize=(6, 8),
                          sharex='all',
                          gridspec_kw={'hspace': 0, 'wspace': 0})

    ax[0].plot(np.array(time_fs), np.array(energies), color="b")
    ax[0].set_ylabel('E (eV/atom)')

    ax[1].plot(np.array(time_fs), temperature, color="r")
    ax[1].set_ylabel('T (K)')

    if max_force_threshold is not None:
        ax[2].plot(np.array(time_fs), max_forces, color="g")
        ax[2].axhline(y=max_force_threshold, color='r', linestyle='--',
                      label='Force threshold')
        ax[2].set_ylabel('Max Force (eV/Å)')
        ax[2].legend()

    ax[-1].set_xlabel('Time (fs)')
    fig.savefig(img_name)
    fig.show()

    # Return simulation statistics
    return {
        'termination_reason': termination_reason,
        'runtime_minutes': runtime_minutes,
        'n_steps_completed': len(time_fs),
        'final_temperature': temperature[-1],
        'final_energy': energies[-1],
        'max_force_reached': max_forces[-1] if max_forces else None
    }

#
# def run_md(dyn, atom_conf, T0, calc, Niter, traj_name, img_name):
#     """
#     Executes a molecular dynamics (MD) simulation and plots the results.
#
#     This function initializes the MD simulation by setting up the atom configuration
#     with a given calculator and assigning an initial temperature using the Maxwell-Boltzmann
#     distribution. It then runs the simulation for a specified number of iterations,
#     logs the simulation data (time, temperature, and potential energy per atom), and
#     plots the energy and temperature profiles over time.
#
#     :param dyn: The molecular dynamics driver to execute the simulation.
#     :type dyn: ase.md.Verlet or other ASE MD driver
#     :param atom_conf: The atomic configuration to be simulated.
#     :type atom_conf: ase.Atoms
#     :param T0: The initial temperature of the system in Kelvin.
#     :type T0: float
#     :param calc: The calculator object to compute forces and energies during the simulation.
#     :type calc: ase.Calculator
#     :param Niter: The number of MD steps to simulate.
#     :type Niter: int
#     :param traj_name: Filename for saving the trajectory of the simulation.
#     :type traj_name: str
#     :param img_name: Filename for saving the energy and temperature plot.
#     :type img_name: str
#     :return: None
#     """
#     def write_frame():
#         dyn.atoms.write(traj_name, append=True)
#         time_fs.append(dyn.get_time() / units.fs)
#         temperature.append(dyn.atoms.get_temperature())
#         energies.append(dyn.atoms.get_potential_energy() / len(dyn.atoms))
#
#     # Initialize
#     random.seed(701)
#     atom_conf.set_calculator(calc)
#     MaxwellBoltzmannDistribution(atom_conf, temperature_K=T0)
#     Stationary(atom_conf)
#     ZeroRotation(atom_conf)
#
#     # Run MD
#     time_fs, temperature, energies = [], [], []
#     dyn.attach(write_frame)
#     t0 = time.time()
#     dyn.run(Niter)
#     t1 = time.time()
#     print("MD finished in {0:.2f} minutes!".format((t1-t0)/60))
#
#     # Plot energy, temperature
#     fig, ax = pl.subplots(2, 1, figsize=(6, 6), sharex='all', gridspec_kw={'hspace': 0, 'wspace': 0})
#     ax[0].plot(np.array(time_fs), np.array(energies), color="b")
#     ax[0].set_ylabel('E (eV/atom)')
#     ax[1].plot(np.array(time_fs), temperature, color="r")
#     ax[1].set_ylabel('T (K)')
#     ax[1].set_xlabel('Time (fs)')
#     fig.savefig(img_name)
#     fig.show()


if __name__ == "__main__":
    temp = 353
    mace_calc_R1 = MACECalculator(model_paths=['../random_model_1.model'], device='cuda')
    mace_calc_R2 = MACECalculator(model_paths=['../random_model_2.model'], device='cuda')
    mace25 = MACECalculator(model_paths=['../mace25.model'], device='cuda')
    rattled50 = MACECalculator(model_paths=['../rattled50.model'], device='cuda')
    rattled100 = MACECalculator(model_paths=['../rattled-mace.model'], device='cuda')
    init_conf = ea.sel_by_info_val(
        read('../structure_forces.xyz', ':'),
        'nneightol',
        1.2)[0].copy()

    # Get cutoff
    cutoff = natural_cutoffs(init_conf)
    # Si = 1.11; O = 0.66; H = 0.31; C = 0.76; Zr = 1.75

    # Nose-Hoover thermostat
    nose = NPT(
        atoms=init_conf,
        timestep=0.5*units.fs,
        temperature_K=temp,
        ttime=15*units.fs,
        externalstress=None,  # No stress for NVT
        logfile='md_nose.log'
        #trajectory='nose_353K.traj',
        #loginterval=1  # Log every step
    )

    # Langevin thermostat
    langevin = Langevin(
        atoms=init_conf,
        timestep=0.5*units.fs,
        temperature_K=temp,
        friction=0.1,
        logfile='md_langevin.log'
    )

    # Run MD
    stats = run_md(
        dyn=nose,
        atom_conf=init_conf,
        T0=300,
        calc=rattled100,
        Niter=10000,
        traj_name='rattle_f10.xyz',
        img_name='rattle_f10.png',
        max_force_threshold=10.0
    )

    print(f"Simulation {stats['termination_reason']}")
    print(f"Completed {stats['n_steps_completed']} steps")
    print(f"Maximum force reached: {stats['max_force_reached']:.2f} eV/Å")

    # stats2 = run_md(
    #     dyn=nose,
    #     atom_conf=init_conf,
    #     T0=300,
    #     calc=rattled50,
    #     Niter=10000,
    #     traj_name='rattle50_f10.xyz',
    #     img_name='rattle50_f10.png',
    #     max_force_threshold=10.0
    # )
    #
    # print(f"Simulation {stats2['termination_reason']}")
    # print(f"Completed {stats2['n_steps_completed']} steps")
    # print(f"Maximum force reached: {stats2['max_force_reached']:.2f} eV/Å")
    #
    # stats3 = run_md(
    #     dyn=nose,
    #     atom_conf=init_conf,
    #     T0=300,
    #     calc=rattled100,
    #     Niter=10000,
    #     traj_name='rattle100_f10.xyz',
    #     img_name='rattle100_f10.png',
    #     max_force_threshold=10.0
    # )
    #
    # print(f"Simulation {stats3['termination_reason']}")
    # print(f"Completed {stats3['n_steps_completed']} steps")
    # print(f"Maximum force reached: {stats3['max_force_reached']:.2f} eV/Å")

    # run_md(
    #     dyn=nose,
    #     atom_conf=init_conf,
    #     T0=300,
    #     calc=mace_calc_R2,
    #     Niter=10000,
    #     traj_name='nose_353K_R2_300.xyz',
    #     img_name='nose_353K_R2_300.png',
    # )

    # run_md(
    #     dyn=langevin,
    #     atom_conf=init_conf,
    #     T0=300,
    #     calc=mace_calc,
    #     Niter=10,
    #     traj_name='nose_353K.traj',
    #     img_name='nose_353K.png'
    # )