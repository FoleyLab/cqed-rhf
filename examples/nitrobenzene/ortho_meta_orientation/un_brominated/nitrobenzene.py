import numpy as np
import psi4

ortho_string = """
0 1
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
units angstrom
no_reorient
no_com
symmetry c1
"""

# ----------------------------
# Psi4 options
# ----------------------------
psi4_options = {
    "basis": "6-311G*",
    "scf_type": "df",          # density fitting
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}


psi4.set_memory("24 GB")
psi4.core.set_output_file("psi4_ortho_unbrominated_cation.out", True)
psi4.set_options(psi4_options)
mol = psi4.geometry(ortho_string)


# Perform the geometry optimization using HF/STO-3G
energy = psi4.optimize('WB97X-D')

# Print the final optimized geometry
print("Final optimized geometry (Angstroms):")
mol.print_out_in_angstrom()

