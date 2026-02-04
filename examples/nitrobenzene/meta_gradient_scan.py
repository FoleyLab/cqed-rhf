import numpy as np
import psi4

from cqed_rhf import CQEDRHFCalculator

meta_coords = [
 "C                  0.02949981    1.33972592    0.06817723",
 "C                  1.43483278    1.28667967    0.00635313",
 "C                  2.11179024    0.05106117   -0.00544138",
 "C                  1.44506636   -1.13720058    0.03116583",
 "C                 -0.68793171    0.16822220    0.10995314",
 "H                 -0.47126997    2.29839666    0.07811355",
 "H                  2.02732783    2.19651728   -0.03220624",
 "H                  1.98966526   -2.07643217    0.02318494",
 "H                 -1.77163480    0.18040547    0.15819632",
 "N                  3.58635895    0.05097292   -0.06745286",
 "O                  4.14711759   -1.05966097   -0.08807849",
 "O                  4.14497859    1.16390951   -0.09010823",
 "C                 -0.02361177   -1.14582791    0.08353483",
 "H                 -0.43674996   -1.87247364    0.78889576",
 "Br                -0.53591638   -1.86972195   -1.74078671"
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
    geometry = make_geometry(meta_coords)

    field_vectors = {
        "z_pol":  [0.0, 0.0, 0.1],
        "x_pol":  [0.1, 0.0, 0.0],
        "y_pol":  [0.0, 0.1, 0.0],
        "diag":   [0.078, 0.055, 0.027],
    }

    basis_sets = ["6-31G", "6-311G*"]

    for basis in basis_sets:
        print(f"\n===== META | basis = {basis} =====")

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

