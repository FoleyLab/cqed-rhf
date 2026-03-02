import numpy as np
import psi4

ortho_string = """
0 1
         C           -1.885946870148     1.189583649403    -0.119726128149
         C           -0.498436150781     1.207756989720    -0.095587674731
         C            0.177022900609     0.001244416404     0.003170781601
         C           -2.570320810975    -0.018930814056    -0.046076352491
         H           -2.433490753919     2.122014686561    -0.196502462215
         H            0.061742696521     2.131508866859    -0.151149504465
         H           -3.654760225882    -0.026965279238    -0.065539508188 
         N            1.653234989788     0.012187306043     0.029557107619  
         O            2.221533450190    -1.057060327273     0.115433821878  
         O            2.208198822578     1.089819744626    -0.036210769804    
         C           -1.871287950510    -1.217279026698     0.052462992898    
         H           -2.407539508613    -2.157651159151     0.109753013519    
         C           -0.483684066407    -1.215092211065     0.078168902711    
         H            0.087632738087    -2.130465296765     0.154596572701 
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


# Perform the geometry optimization wb97x-D/6-311G*
energy = psi4.optimize('WB97X-D')

# Print the final optimized geometry
print("Final optimized geometry (Angstroms):")
mol.print_out_in_angstrom()

