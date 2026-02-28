import numpy as np
import psi4
psi4.core.be_quiet()
from cqed_rhf import CQEDRHFCalculator

para_coords = [
    "C      -0.51161830      1.24438602      0.73214005",
    "C       0.85650059      1.25190371      0.71794822",
    "C       1.52411872      0.02466192      0.71392779",
    "H      -1.07180440      2.17268231      0.74592571",
    "H       1.43612896      2.16392187      0.71209901",
    "N       3.00853958      0.04609710      0.69882380",
    "O       3.57509730     -1.08276817      0.69917471",
    "O       3.54211436      1.19087085      0.68920202",
    "C      -0.47546495     -1.25340277      0.74211864",
    "H      -1.00857443     -2.19737796      0.76294579",
    "C       0.89222770     -1.22140781      0.72806582",
    "H       1.49804842     -2.11624470      0.72965321",
    "C      -1.26790658     -0.01584181      0.71212742",
    "H      -2.11652080     -0.02529322      1.40316150",
    "Br     -2.11496699     -0.03466693     -1.12187408"
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

#    field_vectors = {
#        "z_pol":  [0.0, 0.0, 0.1],
#        "x_pol":  [0.1, 0.0, 0.0],
#        "y_pol":  [0.0, 0.1, 0.0],
#        "diag":   [0.078, 0.055, 0.027],
#    }
    field_vectors = {"diag": [0.078, 0.055, 0.027]}
    basis_sets = ["sto-3g", "6-31G"]

    for basis in basis_sets:
        print(f"\n===== PARA | basis = {basis} =====")

        psi4_options = {
            "basis": basis,
            "scf_type": "pk",
            "e_convergence": 1e-12,
            "d_convergence": 1e-12,
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

