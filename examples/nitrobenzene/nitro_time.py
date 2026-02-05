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
    "basis": "6-31G",
    "scf_type": "pk",
    "save_jk": True,          # density fitting (matches intent of old example)
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
# Geometry: nitrobenzene (angstrom)
# =============================================================================

mol_string = """
C -0.80658313  1.22973465  0.03041801
C  0.56153576  1.23725234  0.01622618
C  1.22915389  0.01001055  0.01220575
H -1.36676923  2.15803094  0.04420367
H  1.14116413  2.14927050  0.01037697
N  2.71357475  0.03144573 -0.00289824
O  3.28013247 -1.09741954 -0.00254733
O  3.24714953  1.17621948 -0.01252002
C -0.77042978 -1.26805414  0.04039660
H -1.30353926 -2.21202933  0.06122375
C  0.59726287 -1.23605918  0.02634378
H  1.20308359 -2.13089607  0.02793117
C -1.56287141 -0.03049318  0.01040538
H -2.41148563 -0.03994459  0.70143946
Br -2.40993182 -0.04931830 -1.82359612
1 1
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

