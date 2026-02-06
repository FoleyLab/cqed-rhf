import psi4
import numpy as np
psi4.core.be_quiet()
# =========================
# Initial geometry (angstrom)
# =========================

h2o_string = """
O  0.000000000000   0.000000000000  -0.100000000000
H  0.000000000000  -0.790689573744   0.543701060715
H  0.000000000000   0.790689573744   0.543701060715
units angstrom
no_reorient
no_com
symmetry c1
"""

# =========================
# Psi4 options
# =========================

psi4_options = {
    "basis": "cc-pVDZ",
    "scf_type": "df",
    "e_convergence": 1e-10,
    "d_convergence": 1e-10,
    "geom_maxiter": 50,
    "g_convergence" : "gau_verytight",
}

psi4.set_options(psi4_options)

# =========================
# Run Psi4 geometry optimization
# =========================

mol_ref = psi4.geometry(h2o_string)

E_ref, wfn_ref = psi4.optimize(
    "scf",
    molecule=mol_ref,
    return_wfn=True,
)

# =========================
# Extract optimized geometry
# =========================

coords_ref_bohr = mol_ref.geometry().to_array()
coords_ref_angstrom = coords_ref_bohr * psi4.constants.bohr2angstroms

symbols_ref = [mol_ref.symbol(i) for i in range(mol_ref.natom())]

print("Psi4 RHF optimized energy (Ha):")
print(f"{E_ref:.10f}\n")

print("Psi4 RHF optimized geometry (angstrom):")
for s, (x, y, z) in zip(symbols_ref, coords_ref_angstrom):
    print(f"{s:2s} {x: .8f} {y: .8f} {z: .8f}")


from cqed_rhf import CQEDRHFCalculator
from cqed_rhf.drivers import bfgs_optimize
from cqed_rhf.utils import ANGSTROM_TO_BOHR

# =========================
# CQED parameters (no cavity)
# =========================

lambda_vector = [0., 0., 0.]
omega = 0.1  # irrelevant when lambda = 0

calc = CQEDRHFCalculator(
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True
)

# =========================
# Run CQED-RHF optimization
# =========================

opt_cqed = bfgs_optimize(
    calculator=calc,
    geometry=h2o_string,
    canonical="psi4",   # exact gradients for clean comparison
    gtol=1e-6,
    maxiter=50,
    debug=False,
)

coords_cqed_bohr = opt_cqed.x.reshape(-1, 3)
coords_cqed_angstrom = coords_cqed_bohr / ANGSTROM_TO_BOHR

print("CQED-RHF (λ=0) optimized energy (Ha):")
print(f"{opt_cqed.fun:.10f}\n")

print("CQED-RHF (λ=0) optimized geometry (angstrom):")
for s, (x, y, z) in zip(symbols_ref, coords_cqed_angstrom):
    print(f"{s:2s} {x: .8f} {y: .8f} {z: .8f}")


# =========================
# Compare energies
# =========================

print("Energy difference (CQED - Psi4) [Ha]:")
print(f"{opt_cqed.fun - E_ref:.3e}\n")

# =========================
# Compare geometries
# =========================

geom_diff = coords_cqed_angstrom - coords_ref_angstrom

print("Max geometry difference (angstrom):")
print(f"{np.max(np.abs(geom_diff)):.3e}\n")

print("RMS geometry difference (angstrom):")
print(f"{np.sqrt(np.mean(geom_diff**2)):.3e}")

