import psi4

mol = psi4.geometry("""
O
H 1 0.96
H 1 0.96 2 104.5
symmetry c1
""")

# Psi4 options
psi4.set_options({'basis': 'aug-cc-pvdz','scf_type': 'df','e_convergence': 1e-10,'d_convergence': 1e-10})


e_scf = psi4.energy("scf")
