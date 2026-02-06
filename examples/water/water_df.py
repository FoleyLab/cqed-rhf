import numpy as np
import psi4
import pytest

from cqed_rhf import CQEDRHFCalculator
from cqed_rhf.utils import ANGSTROM_TO_BOHR


# =============================================================================
# Common test data
# =============================================================================
EXPECTED_RHF_E = -76.04123648668632

H2O_GEOM = """
O            0.000000000000     0.000000000000    -0.065775570547
H            0.000000000000    -0.759061990794     0.521953018286
H            0.000000000000     0.759061990794     0.521953018286
units angstrom
no_reorient
no_com
symmetry c1
"""

PSI4_OPTIONS_DF = {
    "basis": "6-31G",
    "scf_type": "df",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}

psi4.set_options(PSI4_OPTIONS_DF)

mol = psi4.geometry(H2O_GEOM)

# Psi4 DF-RHF energy
e_psi4 = psi4.energy("scf")

# CQED-RHF with lambda = 0
calc = CQEDRHFCalculator(
    lambda_vector=[0.0, 0.0, 0.0],
    omega=0.0,
    psi4_options=PSI4_OPTIONS_DF,
    density_fitting=True,
)

e_cqed, _, _ = calc.energy_and_gradient(
    H2O_GEOM,
    canonical="psi4",
)
print(F"Difference between e_cqed and expected: {(e_cqed - e_psi4):12.16e}")
print(F"Difference between psi4   and expected: {(e_psi4 - e_cqed):12.16e}")


