import numpy as np
import psi4
psi4.core.be_quiet()
from cqed_rhf import CQEDRHFCalculator


ortho_string = """
C           -1.804928163307     1.957993763262     0.703312273806
C           -0.379708783307     1.994122833262     0.698532703806
C            0.296125016693     0.817793533262     0.710271493806
C           -2.520286433307     0.755089873262     0.736288843806
H           -2.344947113307     2.899196893262     0.691895063806
H            0.158564066693     2.933869823262     0.699142733806
H           -3.601954283307     0.764862203262     0.746931053806
N            1.767881836693     0.820900013262     0.771891313806
O            2.315054046693    -0.296733496738     0.879853723806
O            2.340645916693     1.923356243262     0.711986073806
C           -1.829967733307    -0.442167236738     0.756258983806
H           -2.356763623307    -1.389967436738     0.789740873806
C           -0.361341153307    -0.491572936738     0.714148383806
H            0.119338216693    -1.238105076738     1.350400383806
BR          -0.151212663307    -1.224162306738    -1.170925976194
1 1
units angstrom
no_reorient
no_com
symmetry c1
"""

ortho_coords = [
 "C                  0.51932475    1.23303451   -0.03194925",
 "C                  1.94454413    1.26916358   -0.03672882",
 "C                  2.62037793    0.09283428   -0.02499003",
 "C                 -0.19603352    0.03013062    0.00102732",
 "H                 -0.02069420    2.17423764   -0.04336646",
 "H                  2.48281698    2.20891057   -0.03611879",
 "H                 -1.27770137    0.03990295    0.01166953",
 "N                  4.09213475    0.09594076    0.03662979",
 "O                  4.63930696   -1.02169275    0.14459220",
 "O                  4.66489883    1.19839699   -0.02327545",
 "C                  0.49428518   -1.16712649    0.02099746",
 "H                 -0.03251071   -2.11492669    0.05447935",
 "C                  1.96291176   -1.21653219   -0.02111314",
 "H                  2.44359113   -1.96306433    0.61513886",
 "Br                 2.17304025   -1.94912156   -1.90618750"
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
    geometry = make_geometry(ortho_coords)

    field_vectors = {
        "z_pol":  [0.0, 0.0, 0.1],
        "x_pol":  [0.1, 0.0, 0.0],
        "y_pol":  [0.0, 0.1, 0.0],
        "diag":   [0.078, 0.055, 0.027],
    }
    basis_sets = ["6-31G*", "6-311G*"]

    for basis in basis_sets:
        print(f"\n===== ORTHO | basis = {basis} =====")

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
                density_fitting=True
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

