import numpy as np
import qutip
import matplotlib.pyplot as plt
import h5py

# Time units in ns, frequency units in GHz
 
# Define simulation and system parameters
fock_trunc = 10
sim_timestep = 3
det = 137.7e-3 # initial cavity-qubit detuning
g = 6.65e-3 # coupling factor

# Define mode operators
a = qutip.tensor(qutip.qeye(fock_trunc), qutip.destroy(2)) # qubit mode
a_dag = a.dag()
b = qutip.tensor(qutip.destroy(fock_trunc), qutip.qeye(2)) # cav mode
b_dag = b.dag()
Pg = qutip.tensor(qutip.qeye(fock_trunc), 
                  qutip.basis(2, 0)*qutip.basis(2, 0).dag()) # Projection

# Initial state: qubit in |e>, cavity in |0>
initial_state = qutip.tensor(qutip.basis(fock_trunc, 0), qutip.basis(2, 1))

# Hamiltonians
H_coupling = 2*np.pi*g*(a_dag*b + b_dag*a)
H_cav = 2*np.pi*det*b_dag*b
H_qubit = lambda f: 2*np.pi*f*a_dag*a # tunable frequency

# Simulate interaction as a function of transmon frequency
total_time = 400
time_list = np.arange(1, total_time, sim_timestep) 

def vacuum_rabi_result(frequency, time_list):
    H = H_cav + H_qubit(frequency) + H_coupling
    return qutip.mesolve(H, initial_state, time_list).states

# Get vacuum rabi data
frequency_list = np.arange(det - 26.6e-3, det + 26e-3, 0.5e-3)
vacuum_rabi_data = []
for f in frequency_list:
    states_list = vacuum_rabi_result(f, time_list)
    vacuum_rabi_contrast = [(x*x.dag()*Pg).tr() for x in states_list]
    vacuum_rabi_data.append(vacuum_rabi_contrast)
vacuum_rabi_data = np.array(vacuum_rabi_data)

# Get experimental data
fname = "104524_somerset_vacuum_rabi_data.h5"

with h5py.File(fname, 'r') as hdf_file:
    x = np.array(hdf_file["x"])
    y = np.array(hdf_file["y"])
    z = np.array(hdf_file["avg_states"])

# Plot
fig, ax = plt.subplots(1, 2, sharey=True, figsize = (5.5, 3))
ax[0].pcolormesh((frequency_list - det)*1e3 , time_list, 
                 vacuum_rabi_data.T, shading="auto")
ax[0].set_title("Simulation")
ax[0].set_ylabel("Interaction time (ns)")
ax[0].set_xlabel("Interaction detuning (MHz)")
ax[1].pcolormesh(x, y, -z, shading="auto")
ax[1].set_title("Experiment")
ax[1].set_xlabel("Flux pulse amplitude (a.u.)")

plt.tight_layout()
plt.savefig('vacuum_rabi_comparison.png')
plt.show()
