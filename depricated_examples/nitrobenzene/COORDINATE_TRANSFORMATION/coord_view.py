import numpy as np
import psi4
import nglview as nv

# coordinates for para intermediate
para_coords = [
    "C                  -0.51161830     1.24438602     0.73214005",
    "C                   0.85650059     1.25190371     0.71794822",
    "C                   1.52411872     0.02466192     0.71392779",
    "H                  -1.07180440     2.17268231     0.74592571",
    "H                   1.43612896     2.16392187     0.71209901",
    "N                   3.00853958     0.04609710     0.69882380",
    "O                   3.57509730    -1.08276817     0.69917471",
    "O                   3.54211436     1.19087085     0.68920202",
    "C                  -0.47546495    -1.25340277     0.74211864",
    "H                  -1.00857443    -2.19737796     0.76294579",
    "C                   0.89222770    -1.22140781     0.72806582",
    "H                   1.49804842    -2.11624470     0.72965321",
    "C                  -1.26790658    -0.01584181     0.71212742",
    "H                  -2.11652080    -0.02529322     1.40316150",
    "Br                 -2.11496699    -0.03466693    -1.12187408"
]

# 1. Convert the list of strings into a single block for Psi4
# We add the charge (1) and multiplicity (1) header here
geom_string = "1 1\n" + "\n".join(para_coords) + "\nunits angstrom"

# 2. Create the Psi4 Geometry Object
para_mol = psi4.geometry(geom_string)

# 3. Visualize with NGLView
# We convert the Psi4 molecule to an XYZ-formatted string, 
# which NGLView can parse easily.
view = nv.show_text(para_mol.save_string_xyz())

# Optional: Improve the representation
view.add_ball_and_stick()
view.remove_cartoon() # Removes default ribbon if present

view
