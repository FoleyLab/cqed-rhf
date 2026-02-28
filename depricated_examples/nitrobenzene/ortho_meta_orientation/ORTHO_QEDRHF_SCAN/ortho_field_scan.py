import numpy as np
import psi4

from cqed_rhf.utils import write_xyz
from cqed_rhf.calculator import CQEDRHFCalculator
from cqed_rhf.drivers import velocity_verlet_md
from cqed_rhf.observables.nitrobenzene_orientation import NitrobenzeneOrientation
from cqed_rhf.utils import write_xyz, ANGSTROM_TO_BOHR


def generate_field_vector_from_theta_and_phi(theta, phi):
    """
    Generate a unit field vector from spherical coordinates.

    Parameters:
    -----------
    theta : float
        Polar angle in degrees (0° = +z axis, 90° = xy-plane, 180° = -z axis)
    phi : float
        Azimuthal angle in degrees (0° = +x axis, 90° = +y axis)

    Returns:
    --------
    array : Field vector [x, y, z] as a unit vector

    Spherical to Cartesian conversion:
        x = sin(θ) cos(φ)
        y = sin(θ) sin(φ)
        z = cos(θ)
    """
    # Convert degrees to radians
    theta_rad = np.radians(theta)
    phi_rad = np.radians(phi)

    # Compute Cartesian components
    x = np.sin(theta_rad) * np.cos(phi_rad)
    y = np.sin(theta_rad) * np.sin(phi_rad)
    z = np.cos(theta_rad)

    return np.array([x, y, z])

# Example: Generate field vector
theta = 45  # 45° from z-axis
phi = 45.0    # 30° from x-axis in xy-plane

# compute field vector and print it
field_vector = generate_field_vector_from_theta_and_phi(theta, phi) * 0.1  # scale by field strength
print(f"Field vector for θ={theta}°, φ={phi}°: {field_vector}")

# ----------------------------
# Molecular geometry (ortho)
# ----------------------------
ortho_string = """
1 1
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
 Br                 2.17304025   -1.94912156   -1.90618750
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
psi4.core.set_output_file("psi4_md.out", False)
omega = 0.06615  # cavity frequency in atomic units (corresponding to ~1.8 eV)

# ----------------------------
# Build calculator
# ----------------------------
calculator = CQEDRHFCalculator(
    lambda_vector=field_vector, # molecule_string=ortho_string, #<-- CQEDRHFCalculator doesn't take molecule_string currently?
    psi4_options=psi4_options,
    omega=omega,
    density_fitting=True,
    charge=0,
    multiplicity=1
)

# ----------------------------
# Build orientation tracker
# ----------------------------
# We need initial coords + symbols for setup
mol = psi4.geometry(ortho_string)
psi4.set_options(psi4_options)
# compute psi4 rhf energy for initial geometry
e_rhf_1 = psi4.energy("scf", molecule=mol)
print(f"Initial RHF energy: {e_rhf_1:.6f} Hartree")
# use calculator to run a single point and get initial energy and forces
e_cqed_1 = calculator.energy(ortho_string)
print(f"Initial CQED-RHF energy: {e_cqed_1:.6f} Hartree")


# loop over theta from 0 to 180 and phi from 0 to 360 and compute energy for each orientation in increments of 2 degree
# print theta, phi, x, y, z, energy to a file or stdout in a formatted way
theta_list = np.arange(0, 181, 2)
phi_list = np.arange(0, 361, 2)


# Note that when theta = 0, all phi values give the same field vector (0, 0, 0.1) since the field is pointing along the z-axis. Similarly, when theta = 180, all phi values give the same field vector (0, 0, -0.1). The variation in energy will be more pronounced at intermediate theta values where the field vector changes direction in the xy-plane as phi varies.
# so we don't need to repeat these calculation for all phi values at theta = 0 and theta = 180, we can just compute it once for each of these theta values and print the result. This will save computational time while still capturing the key orientations of the field.
field_vector_theta_0 = generate_field_vector_from_theta_and_phi(0, 0) * 0.1
field_vector_theta_180 = generate_field_vector_from_theta_and_phi(180, 0) * 0.1

# update calculator with field vector for theta = 0 and compute energy
calculator.lambda_vector = field_vector_theta_0
c_cqed_theta_0 = calculator.energy(ortho_string)

# update calculator with field vector for theta = 180 and compute energy
calculator.lambda_vector = field_vector_theta_180
c_cqed_theta_180 = calculator.energy(ortho_string)


# open file for writing
with open("ortho_field_scan_results.txt", "w") as f:
    for theta in theta_list:
        for phi in phi_list:
            field_vector = generate_field_vector_from_theta_and_phi(theta, phi) * 0.1  # scale by field strength

            if theta == 0:
                field_vector = field_vector_theta_0
                e_cqed = c_cqed_theta_0
            elif theta == 180:
                field_vector = field_vector_theta_180
                e_cqed = c_cqed_theta_180
            else:   
                calculator.lambda_vector = field_vector
                e_cqed = calculator.energy(ortho_string)
            print(f"{theta:3f} {phi:3f} {field_vector[0]: .4f} {field_vector[1]: .4f} {field_vector[2]: .4f} {e_cqed:.12f}")
            f.write(f"{theta:3f} {phi:3f} {field_vector[0]: .4f} {field_vector[1]: .4f} {field_vector[2]: .4f} {e_cqed:.12f}\n")

f.close()


