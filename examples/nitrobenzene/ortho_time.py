"""
Single-point CQED-RHF energy + gradient for nitrobenzene
(using Psi4 DF canonical gradients).

This script mirrors the old oo_cqed_rhf example and is intended
for performance comparisons.
"""

import time
import numpy as np
import psi4

from cqed_rhf import CQEDRHFCalculator
psi4.core.be_quiet()

# =============================================================================
# Psi4 setup
# =============================================================================


psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",         # enable density fitting at psi4 level
    "save_jk": True,          
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}

psi4.set_options(psi4_options)


# =============================================================================
# Cavity parameters
# =============================================================================

lambda_vector = np.array(
    [0.7839420737139418,
     0.5571186332860504,
     0.27395921869243256]
) * 0.1

omega = 0.1  # cavity frequency (a.u.)


# =============================================================================
# Geometry: ortho-substituted nitrobenzene (angstrom)
# =============================================================================

mol_string = """
1 1
 C                  0.51932475    1.23303451   -0.03194925
 C                  1.94454413    1.26916358   -0.03672882
 C                  2.62037793    0.09283428   -0.02499003
 C                 -0.19603352    0.03013062    0.00102732
 H                 -0.02069420    2.17423764   -0.04336646
 H                  2.48281698    2.20891057   -0.03611879
 H                 -1.27770137    0.03990295    0.01166953
 N                  4.09213475    0.09594076    0.03662979
 O                  4.63930696   -1.02169275    0.14459220
 O                  4.66489883    1.19839699   -0.02327545
 C                  0.49428518   -1.16712649    0.02099746
 H                 -0.03251071   -2.11492669    0.05447935
 C                  1.96291176   -1.21653219   -0.02111314
 H                  2.44359113   -1.96306433    0.61513886
 Br                 2.17304025   -1.94912156   -1.90618750
units angstrom
no_reorient
no_com
symmetry c1
"""


# =============================================================================
# Reference RHF energy (for sanity check)
# =============================================================================

mol = psi4.geometry(mol_string)

t0 = time.time()
e_rhf = psi4.energy("scf")
print(f"Standard RHF energy: {e_rhf:.12f} Ha")
print(f"RHF wall time:       {time.time() - t0:.3f} s\n")


# =============================================================================
# CQED-RHF calculation
# =============================================================================
t0 = time.time()
calc = CQEDRHFCalculator(
    lambda_vector=lambda_vector,
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True
)
print(f"instantiated in {time.time()-t0:.3f} s\n")

print("=" * 70)
print("CQED-RHF single-point energy + gradient")
print("=" * 70)

t0 = time.time()
E, grad, g = calc.energy_and_gradient(
    mol_string,
    canonical="psi4",   # matches old: use Psi4 canonical gradient
)
dt = time.time() - t0

print(f"CQED-RHF energy:     {E:.12f} Ha")
print(f"‖∇E‖ (Ha/bohr):     {np.linalg.norm(grad):.3e}")
print(f"Total wall time:    {dt:.3f} s")
print("=" * 70)

