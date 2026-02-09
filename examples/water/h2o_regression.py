import time
import numpy as np
import psi4
psi4.core.be_quiet()

from cqed_rhf import CQEDRHFCalculator
from cqed_rhf.utils import (
    build_psi4_geometry,
    finite_difference_gradient,
)

# =========================
# Geometry (exactly as before)
# =========================

symbols = ["O", "H", "H"]

coords = np.array(
    [
        [0.000000000000,  0.000000000000, -0.068516219320],
        [0.000000000000, -0.790689573744,  0.543701060715],
        [0.000000000000,  0.790689573744,  0.543701060715],
    ]
)

geometry = build_psi4_geometry(coords, symbols, units="angstrom")

# =========================
# Psi4 options
# =========================

psi4_options = {
    "basis": "cc-pVDZ",
    "scf_type": "pk",
    "save_jk": True,
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}

# =========================
# Energy checks
# =========================

# λ = 0 → should match Psi4 RHF exactly
lambda_vector = [0.0, 0.0, 0.0]
calc = CQEDRHFCalculator(lambda_vector, psi4_options)

psi4.set_options(psi4_options)
psi4.geometry(geometry)
E_ref = psi4.energy("scf")

E = calc.energy(geometry)

print(f"RHF ENERGY MATCH: {np.isclose(E, E_ref, atol=1e-9)}")

# λ ≠ 0 → validated CQED-RHF energy
lambda_vector = [0.0, 0.0, 0.05]
calc = CQEDRHFCalculator(lambda_vector, psi4_options)

E, _, _ = calc.energy_and_gradient(geometry, canonical="exact")

E_expected = -76.016355284146
print(f"CQED-RHF ENERGY MATCH: {np.isclose(E, E_expected, atol=1e-9)}\n")

# =========================
# Gradients
# =========================

# Exact analytic gradient
exact_start = time.time()
_, grad_exact, _ = calc.energy_and_gradient(
    geometry, canonical="exact"
)
exact_end = time.time()
# Psi4-based analytic gradient
_, grad_psi4, _ = calc.energy_and_gradient(
    geometry, canonical="psi4"
)
psi4_end = time.time()

# Finite-difference gradient
grad_fd = finite_difference_gradient(
    calculator=calc,
    coords_angstrom=coords,
    symbols=symbols,
    delta=1.0e-4,
)
fd_end = time.time()
# =========================
# Output (mirrors your notebook)
# =========================

print(F"Analytical CQED-RHF Gradient (exact, took {exact_end-exact_start} seconds ):\n")
print(grad_exact)

print(F"\nAnalytical CQED-RHF Gradient (psi4, took {psi4_end-exact_end} seconds):\n")
print(grad_psi4)

print(F"\nNumerical CQED-RHF Gradient (took {fd_end-psi4_end} seconds):\n")
print(grad_fd)

diff_exact_fd = grad_exact - grad_fd
diff_psi4_exact = grad_psi4 - grad_exact

print("\n|| exact − FD || = {:.4e}".format(np.linalg.norm(diff_exact_fd)))
print("|| psi4 − exact || = {:.4e}".format(np.linalg.norm(diff_psi4_exact)))

