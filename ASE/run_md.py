from ase.io import read, write
from ase import units
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import Stationary, ZeroRotation, MaxwellBoltzmannDistribution
from aseMolec import extAtoms as ea
from mace.calculators import MACECalculator

import random
import os
import time
import numpy as np
import pylab as pl
from IPython import display

def simpleMD(init_conf, temp, calc, fname, s, T):
    init_conf.set_calculator(calc)

    #initialize the temperature
    random.seed(701) #just making sure the MD failure is reproducible
    MaxwellBoltzmannDistribution(init_conf, temperature_K=300) #initialize temperature at 300
    Stationary(init_conf)
    ZeroRotation(init_conf)

    dyn = Langevin(init_conf, 1.0*units.fs, temperature_K=temp, friction=0.1) #drive system to desired temperature

    time_fs = []
    temperature = []
    energies = []

    #remove previously stored trajectory with the same name
    os.system('rm -rfv '+fname)

    fig, ax = pl.subplots(2, 1, figsize=(6,6), sharex='all', gridspec_kw={'hspace': 0, 'wspace': 0})

    def write_frame():
        dyn.atoms.write(fname, append=True)
        time_fs.append(dyn.get_time()/units.fs)
        temperature.append(dyn.atoms.get_temperature())
        energies.append(dyn.atoms.get_potential_energy()/len(dyn.atoms))

        ax[0].plot(np.array(time_fs), np.array(energies), color="b")
        ax[0].set_ylabel('E (eV/atom)')

        # plot the temperature of the system as subplots
        ax[1].plot(np.array(time_fs), temperature, color="r")
        ax[1].set_ylabel('T (K)')
        ax[1].set_xlabel('Time (fs)')

        display.clear_output(wait=True)
        display.display(pl.gcf())
        time.sleep(0.01)

    dyn.attach(write_frame, interval=s)
    t0 = time.time()
    dyn.run(T)
    t1 = time.time()
    print("MD finished in {0:.2f} minutes!".format((t1-t0)/60))
    fig.savefig("mace_ase_sim.png")
    fig.show()

mace_calc = MACECalculator(model_paths=['../av_adsorption_353K.model'], device='cuda')
init_conf = ea.sel_by_info_val(read('../Forces_and_traj_353_1.xyz',':'), 'nneightol', 1.2)[0].copy()
simpleMD(init_conf, temp=353, calc=mace_calc, fname='mace_md.xyz', s=5, T=5000)