from ase.io import read
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from ASE.aseMolec import extAtoms as ea
from mace.calculators import MACECalculator
from ase import Atoms
from ase.io import read, write
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase.md.npt import NPT
import random
import time
import numpy as np
import pylab as pl

def run_md(dyn, atom_conf, T0, calc, Niter, traj_name, img_name):
    """
    Executes a molecular dynamics (MD) simulation and plots the results.

    This function initializes the MD simulation by setting up the atom configuration
    with a given calculator and assigning an initial temperature using the Maxwell-Boltzmann
    distribution. It then runs the simulation for a specified number of iterations,
    logs the simulation data (time, temperature, and potential energy per atom), and
    plots the energy and temperature profiles over time.

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
    :return: None
    """
    def write_frame():
        dyn.atoms.write(traj_name, append=True)
        time_fs.append(dyn.get_time() / units.fs)
        temperature.append(dyn.atoms.get_temperature())
        energies.append(dyn.atoms.get_potential_energy() / len(dyn.atoms))

    # Initialize
    random.seed(701)
    atom_conf.set_calculator(calc)
    MaxwellBoltzmannDistribution(atom_conf, temperature_K=T0)
    Stationary(atom_conf)
    ZeroRotation(atom_conf)

    # Run MD
    time_fs, temperature, energies = [], [], []
    dyn.attach(write_frame)
    t0 = time.time()
    dyn.run(Niter)
    t1 = time.time()
    print("MD finished in {0:.2f} minutes!".format((t1-t0)/60))

    # Plot energy, temperature
    fig, ax = pl.subplots(2, 1, figsize=(6, 6), sharex='all', gridspec_kw={'hspace': 0, 'wspace': 0})
    ax[0].plot(np.array(time_fs), np.array(energies), color="b")
    ax[0].set_ylabel('E (eV/atom)')
    ax[1].plot(np.array(time_fs), temperature, color="r")
    ax[1].set_ylabel('T (K)')
    ax[1].set_xlabel('Time (fs)')
    fig.savefig(img_name)
    fig.show()


if __name__ == "__main__":
    temp = 353
    mace_calc = MACECalculator(model_paths=['../av_adsorption_353K.model'], device='cuda')
    init_conf = ea.sel_by_info_val(
        read('../structure_forces.xyz', ':'),
        'nneightol',
        1.2)[0].copy()

    # Nose-Hoover thermostat
    nose = NPT(
        atoms=init_conf,
        timestep=0.25*units.fs,
        temperature_K=temp,
        ttime=15*units.fs,
        externalstress=None,  # No stress for NVT
        logfile='md_nose_25.log',
        loginterval=1  # Log every step
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
    run_md(
        dyn=nose,
        atom_conf=init_conf,
        T0=300,
        calc=mace_calc,
        Niter=20000,
        traj_name='nose_353K_25.traj',
        img_name='nose_353K_25.png'
    )
    # run_md(
    #     dyn=langevin,
    #     atom_conf=init_conf,
    #     T0=300,
    #     calc=mace_calc,
    #     Niter=10,
    #     traj_name='nose_353K.traj',
    #     img_name='nose_353K.png'
    # )