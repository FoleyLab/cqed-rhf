import numpy as np
import psi4

from cqed_rhf.utils import write_xyz
from cqed_rhf.calculator import CQEDRHFCalculator
from cqed_rhf.drivers import velocity_verlet_md
from cqed_rhf.observables.nitrobenzene_orientation import NitrobenzeneOrientation
from cqed_rhf.utils import write_xyz, ANGSTROM_TO_BOHR

#<-- Change this to a number of theta and phi values
#<-- you want to compute the scan over!
num_phi_vals = 2
num_theta_vals = 2

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
# Molecular geometry (meta)
# ----------------------------
meta_string = """
1 1
         C           -0.929257263947     2.021527608578     0.744707683350
         C            0.476075706053     1.968481358578     0.682883583350
         C            1.153033166053     0.732862858578     0.67108907335
         C            0.486309286053    -0.455398891422     0.707696283350
         C           -1.646688783947     0.850023888578     0.786483593350
         H           -1.430027043947     2.980198348578     0.754644003350
         H            1.068570756053     2.878318968578     0.644324213350
         H            1.030908186053    -1.394630481422     0.699715393350
         H           -2.730391873947     0.862207158578     0.834726773350
         N            2.627601876053     0.732774608578     0.609077593350
         O            3.188360516053    -0.377859281422     0.588451963350
         O            3.186221516053     1.845711198578     0.586422223350
         C           -0.982368843947    -0.464026221422     0.760065283350
         H           -1.395507033947    -1.190671951422     1.465426213350
         BR          -1.494673453947    -1.187920261422    -1.064256256650
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
mol = psi4.geometry(meta_string)
psi4.set_options(psi4_options)
# compute psi4 rhf energy for initial geometry
e_rhf_1 = psi4.energy("scf", molecule=mol)
print(f"Initial RHF energy: {e_rhf_1:.6f} Hartree")
# use calculator to run a single point and get initial energy and forces
e_cqed_1 = calculator.energy(meta_string)
print(f"Initial CQED-RHF energy: {e_cqed_1:.6f} Hartree")


# loop over theta from 0 to 180 and phi from 0 to 360 and compute energy for each orientation in increments of 2 degree
# print theta, phi, x, y, z, energy to a file or stdout in a formatted way
theta_list = np.arange(0, 181, num_theta_vals)
phi_list = np.arange(0, 361, num_phi_vals)


# Note that when theta = 0, all phi values give the same field vector (0, 0, 0.1) since the field is pointing along the z-axis. Similarly, when theta = 180, all phi values give the same field vector (0, 0, -0.1). The variation in energy will be more pronounced at intermediate theta values where the field vector changes direction in the xy-plane as phi varies.
# so we don't need to repeat these calculation for all phi values at theta = 0 and theta = 180, we can just compute it once for each of these theta values and print the result. This will save computational time while still capturing the key orientations of the field.
field_vector_theta_0 = generate_field_vector_from_theta_and_phi(0, 0) * 0.1
field_vector_theta_180 = generate_field_vector_from_theta_and_phi(180, 0) * 0.1

# update calculator with field vector for theta = 0 and compute energy
calculator.lambda_vector = field_vector_theta_0
c_cqed_theta_0 = calculator.energy(meta_string)

# update calculator with field vector for theta = 180 and compute energy
calculator.lambda_vector = field_vector_theta_180
c_cqed_theta_180 = calculator.energy(meta_string)


# open file for writing
with open("meta_field_scan_results.txt", "w") as f:
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
                e_cqed = calculator.energy(meta_string)
            print(f"{theta:3f} {phi:3f} {field_vector[0]: .4f} {field_vector[1]: .4f} {field_vector[2]: .4f} {e_cqed:.12f}")
            f.write(f"{theta:3f} {phi:3f} {field_vector[0]: .4f} {field_vector[1]: .4f} {field_vector[2]: .4f} {e_cqed:.12f}\n")

f.close()


