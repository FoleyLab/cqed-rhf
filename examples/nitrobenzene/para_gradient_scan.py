import numpy as np
import psi4

from cqed_rhf import CQEDRHFCalculator

para_coords = [
 "C -0.80658313  1.22973465  0.03041801",
 "C  0.56153576  1.23725234  0.01622618",
 "C  1.22915389  0.01001055  0.01220575",
 "H -1.36676923  2.15803094  0.04420367",
 "H  1.14116413  2.14927050  0.01037697",
 "N  2.71357475  0.03144573 -0.00289824",
 "O  3.28013247 -1.09741954 -0.00254733",
 "O  3.24714953  1.17621948 -0.01252002",
 "C -0.77042978 -1.26805414  0.04039660",
 "H -1.30353926 -2.21202933  0.06122375",
 "C  0.59726287 -1.23605918  0.02634378",
 "H  1.20308359 -2.13089607  0.02793117",
 "C -1.56287141 -0.03049318  0.01040538",
 "H -2.41148563 -0.03994459  0.70143946",
 "Br -2.40993182 -0.04931830 -1.82359612",
]

def make_geometry(coords):
    return "\n".join(coords) + """
1 1
units angstrom
no_reorient
no_com
symmetry c1
"""

def run():
    geometry = make_geometry(para_coords)

    field_vectors = {
        "z_pol":  [0.0, 0.0, 0.1],
        "x_pol":  [0.1, 0.0, 0.0],
        "y_pol":  [0.0, 0.1, 0.0],
        "diag":   [0.078, 0.055, 0.027],
    }
    basis_sets = ["6-31G", "6-311G*"]

    for basis in basis_sets:
        print(f"\n===== PARA | basis = {basis} =====")

        psi4_options = {
            "basis": basis,
            "scf_type": "df",
            "e_convergence": 1e-8,
            "d_convergence": 1e-8,
        }

        for label, lam in field_vectors.items():
            calc = CQEDRHFCalculator(
                lambda_vector=lam,
                psi4_options=psi4_options,
                omega=0.1,
            )

            E, grad, _ = calc.energy_and_gradient(
                geometry,
                canonical="psi4",
            )

            print(f"\nField: {label}  lambda={lam}")
            print(f"Energy (Ha): {E:.10f}")
            print(f"|Grad| (Ha/bohr): {np.linalg.norm(grad):.6e}")
            print(grad)

if __name__ == "__main__":
    run()

