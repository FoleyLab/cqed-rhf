import numpy as np
import psi4

from cqed_rhf import CQEDRHFCalculator



meta_coords = [
    "C      -0.92925726      2.02152761      0.74470768",
    "C       0.47607571      1.96848136      0.68288358",
    "C       1.15303317      0.73286286      0.67108907",
    "C       0.48630929     -0.45539889      0.70769628",
    "C      -1.64668878      0.85002389      0.78648359",
    "H      -1.43002704      2.98019835      0.75464400",
    "H       1.06857076      2.87831897      0.64432421",
    "H       1.03090819     -1.39463048      0.69971539",
    "H      -2.73039187      0.86220716      0.83472677",
    "N       2.62760188      0.73277461      0.60907759",
    "O       3.18836052     -0.37785928      0.58845196",
    "O       3.18622152      1.84571120      0.58642222",
    "C      -0.98236884     -0.46402622      0.76006528",
    "H      -1.39550703     -1.19067195      1.46542621",
    "Br     -1.49467345     -1.18792026     -1.06425626"
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
        "diag":   [0.078, 0.055, 0.027],
    }

    basis_sets = ["6-31G", "6-311G*"]

    for basis in basis_sets:
        print(f"\n===== META | basis = {basis} =====")

        psi4_options = {
            "basis": basis,
            "scf_type": "df",
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

