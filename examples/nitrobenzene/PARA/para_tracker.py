import numpy as np
import psi4

from cqed_rhf.utils import write_xyz
from cqed_rhf.calculator import CQEDRHFCalculator
from cqed_rhf.drivers import velocity_verlet_md
from cqed_rhf.observables.nitrobenzene_orientation import NitrobenzeneOrientation
from cqed_rhf.utils import write_xyz, ANGSTROM_TO_BOHR

BOHR_TO_ANG = 0.52917721092
# ----------------------------
# Molecular geometry (para)
# ----------------------------
para_string = """
1 1
         C           -0.511618296797     1.244386024531     0.732140048697
         C            0.856500593203     1.251903714531     0.717948218697
         C            1.524118723203     0.024661924531     0.713927788697
         H           -1.071804396797     2.172682314531     0.745925708697
         H            1.436128963203     2.163921874531     0.712099008697
         N            3.008539583203     0.046097104531     0.698823798697
         O            3.575097303203    -1.082768165469     0.699174708697
         O            3.542114363203     1.190870854531     0.689202018697
         C           -0.475464946797    -1.253402765469     0.742118638697
         H           -1.008574426797    -2.197377955469     0.762945788697
         C            0.892227703203    -1.221407805469     0.728065818697
         H            1.498048423203    -2.116244695469     0.729653208697
         C           -1.267906576797    -0.015841805469     0.712127418697
         H           -2.116520796797    -0.025293215469     1.403161498697
         BR          -2.114966986797    -0.034666925469    -1.121874081303
units angstrom
no_reorient
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
psi4.core.set_output_file("para_md.out", False)


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
mol = psi4.geometry(meta_string)
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
    geometry=para_string,
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


xyz_file = "para.xyz"
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
