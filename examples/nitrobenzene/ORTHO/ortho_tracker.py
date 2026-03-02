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
C           -1.804928163307     1.957993763262     0.703312273806
C           -0.379708783307     1.994122833262     0.698532703806
C            0.296125016693     0.817793533262     0.710271493806
C           -2.520286433307     0.755089873262     0.736288843806
H           -2.344947113307     2.899196893262     0.691895063806
H            0.158564066693     2.933869823262     0.699142733806
H           -3.601954283307     0.764862203262     0.746931053806
N            1.767881836693     0.820900013262     0.771891313806
O            2.315054046693    -0.296733496738     0.879853723806
O            2.340645916693     1.923356243262     0.711986073806
C           -1.829967733307    -0.442167236738     0.756258983806
H           -2.356763623307    -1.389967436738     0.789740873806
C           -0.361341153307    -0.491572936738     0.714148383806
H            0.119338216693    -1.238105076738     1.350400383806
BR          -0.151212663307    -1.224162306738    -1.170925976194
1 1
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


xyz_file = "ortho.xyz"
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
