import numpy as np
import psi4

from cqed_rhf import CQEDRHFCalculator
from cqed_rhf.observables.nitrobenzene_orientation import NitrobenzeneOrientation
from cqed_rhf.observables.torque_tracker import RotationalProjectionObserver


# -----------------------------
# External CAS data
# -----------------------------
casscf_energy = -3006.175906038858

casscf_gradient = np.array([
    [-0.0001034324, -0.0008398526,  0.0014513293],
    [ 0.0034629473, -0.0095585074, -0.0003296254],
    [-0.0085594373,  0.0112013503,  0.0017401134],
    [-0.0137195401,  0.0000224905, -0.0070515244],
    [ 0.0052736515, -0.0068028852, -0.0064851842],
    [-0.0043250856,  0.0081589988,  0.0004064198],
    [ 0.0095719988,  0.0117220665, -0.0002172812],
    [ 0.0062721695, -0.0103461045,  0.0000228443],
    [-0.0134444035, -0.0015034951,  0.0000910199],
    [-0.0853028254,  0.0045596054,  0.0021065667],
    [ 0.0478107883, -0.1015287416, -0.0026080484],
    [ 0.0562974532,  0.0988566062,  0.0005015843],
    [ 0.0020311329,  0.0046770010,  0.0005309231],
    [-0.0060276975, -0.0101919872,  0.0079701152],
    [ 0.0007622801,  0.0015734547,  0.0018707475]
])


# -----------------------------
# Geometry
# -----------------------------
meta_coords = [
 "C 0.02949981 1.33972592 0.06817723",
 "C 1.43483278 1.28667967 0.00635313",
 "C 2.11179024 0.05106117 -0.00544138",
 "C 1.44506636 -1.13720058 0.03116583",
 "C -0.68793171 0.16822220 0.10995314",
 "H -0.47126997 2.29839666 0.07811355",
 "H 2.02732783 2.19651728 -0.03220624",
 "H 1.98966526 -2.07643217 0.02318494",
 "H -1.77163480 0.18040547 0.15819632",
 "N 3.58635895 0.05097292 -0.06745286",
 "O 4.14711759 -1.05966097 -0.08807849",
 "O 4.14497859 1.16390951 -0.09010823",
 "C -0.02361177 -1.14582791 0.08353483",
 "H -0.43674996 -1.87247364 0.78889576",
 "Br -0.53591638 -1.86972195 -1.74078671"
]


def make_geometry(coords):
    return "\n".join(coords) + """
1 1
units angstrom
no_reorient
no_com
symmetry c1
"""


# -----------------------------
# Main comparison
# -----------------------------
def run():

    geometry = make_geometry(meta_coords)
    field_vector = np.array([0.078, 0.055, 0.027])

    psi4_options = {
        "basis": "6-311G*",
        "scf_type": "df",
        "e_convergence": 1e-12,
        "d_convergence": 1e-12,
    }

    calc = CQEDRHFCalculator(
        lambda_vector=field_vector,
        #molecule_string=geometry,
        psi4_options=psi4_options,
        omega=0.1,
        density_fitting=True,
        charge=1,
        multiplicity=1
    )


    # ---- QED-RHF calculation ----
    E, grad_qed, _ = calc.energy_and_gradient(
        geometry,
        canonical="psi4",
    )

    # ---- Build orientation + projection tools ----
    mol = psi4.geometry(geometry)
    coords_bohr = mol.geometry().to_array()
    symbols = [mol.symbol(i) for i in range(mol.natom())]
    masses = np.array([mol.mass(i) for i in range(mol.natom())])

    orientation = NitrobenzeneOrientation(
        symbols=symbols,
        coords_bohr=coords_bohr,
        field_vector=field_vector,
    )

    projector = RotationalProjectionObserver(
        orientation_tracker=orientation,
        masses=masses,
    )

    # ---- Rotational projections ----
    qed_rot = projector.observe(coords_bohr, grad_qed)
    cas_rot = projector.observe(coords_bohr, casscf_gradient)

    # ---- Output ----
    print("\n================ ENERGY COMPARISON ================")
    print(f"QED-RHF Energy (Ha): {E:.10f}")
    print(f"CAS Energy (Ha):     {casscf_energy:.10f}")
    print(f"Energy Difference:   {E - casscf_energy:.10f}")

    print("\n================ GRADIENT NORMS ===================")
    print(f"QED-RHF |grad|: {np.linalg.norm(grad_qed):.6e}")
    print(f"CAS     |grad|: {np.linalg.norm(casscf_gradient):.6e}")
    print(f"Difference norm: {np.linalg.norm(grad_qed - casscf_gradient):.6e}")

    print("\n================ ROTATIONAL PROJECTIONS ===========")
    print("                QED-RHF            CAS")
    print(f"dE/dphi     {qed_rot['dE_dphi']: .6e}   {cas_rot['dE_dphi']: .6e}")
    print(f"dE/dtheta   {qed_rot['dE_dtheta']: .6e}   {cas_rot['dE_dtheta']: .6e}")

    print("\nDifference in rotational components:")
    print(f"Δ(dE/dphi)   = {qed_rot['dE_dphi'] - cas_rot['dE_dphi']: .6e}")
    print(f"Δ(dE/dtheta) = {qed_rot['dE_dtheta'] - cas_rot['dE_dtheta']: .6e}")


if __name__ == "__main__":
    run()

