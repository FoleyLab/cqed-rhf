import numpy as np
import psi4

from cqed_rhf.utils import write_xyz
from cqed_rhf.calculator import CQEDRHFCalculator
from cqed_rhf.drivers import velocity_verlet_md
from cqed_rhf.observables.nitrobenzene_orientation import NitrobenzeneOrientation
from cqed_rhf.utils import write_xyz, ANGSTROM_TO_BOHR

BOHR_TO_ANG = 0.52917721092
# ----------------------------
# Molecular geometry (meta)
# ----------------------------
meta_string = """
1 1
         C           -0.929257263947     2.021527608578     0.744707683350
         C            0.476075706053     1.968481358578     0.682883583350
         C            1.153033166053     0.732862858578     0.67108907335
         C            0.486309286053    -0.455398891422     0.707696283350
         C           -1.646688783947     0.850023888578     0.786483593350
         H           -1.430027043947     2.980198348578     0.754644003350
         H            1.068570756053     2.878318968578     0.644324213350
         H            1.030908186053    -1.394630481422     0.699715393350
         H           -2.730391873947     0.862207158578     0.834726773350
         N            2.627601876053     0.732774608578     0.609077593350
         O            3.188360516053    -0.377859281422     0.588451963350
         O            3.186221516053     1.845711198578     0.586422223350
         C           -0.982368843947    -0.464026221422     0.760065283350
         H           -1.395507033947    -1.190671951422     1.465426213350
         BR          -1.494673453947    -1.187920261422    -1.064256256650
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
    geometry=meta_string,
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


xyz_file = "meta.xyz"
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
