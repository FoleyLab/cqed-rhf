import numpy as np
from cqed_rhf import CQEDRHFCalculator
from cqed_rhf.utils import build_psi4_geometry

psi4_options = {
    "basis": "cc-pVDZ",
    "scf_type": "df",
    "e_convergence": 1e-8,
    "d_convergence": 1e-6,
}


symbols = ["O", "H", "H"]


coords = [
    [0.000000, 0.000000, 0.000000],
    [0.000000, 0.000000, 1.809],
    [1.751, 0.000000, -0.453],
]

lambda_vec = np.array([0, 0, 0.05])

geom = build_psi4_geometry(coords, symbols, units="bohr")

calc = CQEDRHFCalculator(lambda_vec, psi4_options)
E, grad, g = calc.energy_and_gradient(geom)

