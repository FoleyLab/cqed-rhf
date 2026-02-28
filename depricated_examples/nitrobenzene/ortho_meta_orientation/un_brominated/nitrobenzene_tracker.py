import numpy as np
import psi4

from cqed_rhf.utils import write_xyz
from cqed_rhf.calculator import CQEDRHFCalculator
from cqed_rhf.drivers import velocity_verlet_md
from cqed_rhf.observables.nitrobenzene_orientation import NitrobenzeneOrientation
from cqed_rhf.utils import write_xyz, ANGSTROM_TO_BOHR

BOHR_TO_ANG = 0.52917721092
# ----------------------------
# Molecular geometry (ortho)
# ----------------------------
ortho_string = """
0 1
    C            0.523578368385     1.255907785448    -0.060108858355
    C            1.911089087752     1.274081125765    -0.035970404937
    C            2.586548139142     0.067568552449     0.062788051395
    C           -0.160795572442     0.047393321989     0.013540917303
    H           -0.023965515386     2.188338822606    -0.136885192421
    H            2.471267935054     2.197833002904    -0.091532234671
    H           -1.245234987349     0.039358856807    -0.005922238394
    N            4.062760228321     0.078511442088     0.089174377413
    O            4.631058688723    -0.990736191228     0.175051091672
    O            4.617724061111     1.156143880671     0.023406499990
    C            0.538237288023    -1.150954890653     0.112080262692
    H            0.001985729920    -2.091327023106     0.169370283313
    C            1.925841172126    -1.148768075020     0.137786172505
    H            2.497157976620    -2.064141160720     0.214213842495
units angstrom
no_reorient
no_com
symmetry c1
"""


# ----------------------------
# Cavity / field parameters
# ----------------------------
field_vector = np.array([0.078, 0.055, 0.027])
omega = 0.06615  


# ----------------------------
# Psi4 options
# ----------------------------
psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",          # density fitting
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}

psi4.set_memory("24 GB")
psi4.core.set_output_file("psi4_md.out", False)


# ----------------------------
# Build calculator
# ----------------------------
calculator = CQEDRHFCalculator(
    lambda_vector=field_vector, # molecule_string=ortho_string, #<-- CQEDRHFCalculator doesn't take molecule_string currently?
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    charge=0,
    multiplicity=1
)


# ----------------------------
# Build orientation tracker
# ----------------------------
# We need initial coords + symbols for setup
mol = psi4.geometry(ortho_string)
symbols = [mol.symbol(i) for i in range(mol.natom())]
coords_bohr = mol.geometry().to_array()

orientation_tracker = NitrobenzeneOrientation(
    symbols=symbols,
    coords_bohr=coords_bohr,
    field_vector=field_vector,
)


# ----------------------------
# Run MD
# ----------------------------
traj, observer_data = velocity_verlet_md(
    calculator=calculator,
    geometry=ortho_string,
    dt=10.0,              # atomic units
    nsteps=500,
    canonical="psi4",
    observers=[orientation_tracker],
    debug=True,
)


# ----------------------------
# Inspect results
# ----------------------------
orientation_history = observer_data[orientation_tracker]

phi = np.array([d["phi_deg"] for d in orientation_history])
theta = np.array([d["theta_deg"] for d in orientation_history])

for i in range(len(phi)):
    
    print(f" phi   at step {i} is {phi[i]:.2f} deg")
    print(f" theta at step {i} is {theta[i]:.2f} deg")

print("\nFinal orientation:")
print(f"  phi   = {phi[-1]:.2f} deg")
print(f"  theta = {theta[-1]:.2f} deg")


xyz_file = "nitrobenzene.xyz"
# write coords and theta and phi to trajectory file
for i, frame in enumerate(traj):

    write_xyz(
        filename=xyz_file,
        symbols=symbols,
        coords_angstrom=frame["coords"],
        comment=(
            f"Step {frame['step']}  "
            f"E={frame['energy']:.10f}  "
            f"phi={phi[i]:.3f}  "
            f"theta={theta[i]:.3f}"
        ),
        mode="w" if i == 0 else "a",
    )
