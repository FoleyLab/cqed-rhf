import numpy as np

# =========================
# Unit conversions
# =========================

ANGSTROM_TO_BOHR = 1.0 / 0.52917721092
BOHR_TO_ANGSTROM = 0.52917721092
AMU_TO_AU = 1822.888486209
AU_TO_FS = 0.024188842543
FS_TO_AU = 1.0 / AU_TO_FS



def angstrom_to_bohr(x):
    return x * ANGSTROM_TO_BOHR


def bohr_to_angstrom(x):
    return x * BOHR_TO_ANGSTROM


# =========================
# Geometry builders
# =========================

def build_psi4_geometry(coords, symbols, units="angstrom", symmetry="c1"):
    """
    Build a Psi4 geometry string from coordinates.

    Parameters
    ----------
    coords : ndarray, shape (N, 3)
        Cartesian coordinates.
    symbols : list[str]
        Atomic symbols.
    units : {'bohr', 'angstrom'}
        Units of coords.
    symmetry : str
        Psi4 symmetry setting (default c1).

    Returns
    -------
    geometry : str
        Psi4 geometry string.
    """
    coords = np.asarray(coords)

    if units.lower() == "bohr":
        coords = bohr_to_angstrom(coords)

    lines = []
    for sym, (x, y, z) in zip(symbols, coords):
        lines.append(f"{sym} {x:.12f} {y:.12f} {z:.12f}")
    lines.append("1 1") # charge and multiplicity (default to singlet)
    lines.append("units angstrom")
    lines.append("no_reorient")
    lines.append("no_com")
    lines.append(f"symmetry {symmetry}")

    return "\n".join(lines)


def parse_psi4_geometry(geometry):
    """
    Extract symbols and coordinates from a Psi4 geometry string.

    Returns
    -------
    symbols : list[str]
    coords_bohr : ndarray, shape (N, 3)
    """
    symbols = []
    coords = []

    for line in geometry.splitlines():
        parts = line.strip().split()
        if len(parts) == 4:
            try:
                x, y, z = map(float, parts[1:])
                symbols.append(parts[0])
                coords.append([x, y, z])
            except ValueError:
                pass

    return symbols, np.asarray(coords)


# =========================
# Optimizer / MD helpers
# =========================

def flatten_coords(coords):
    """(N,3) → (3N,)"""
    return np.asarray(coords).reshape(-1)


def reshape_coords(x):
    """(3N,) → (N,3)"""
    x = np.asarray(x)
    return x.reshape(-1, 3)


def kinetic_energy(velocities, masses):
    """
    Classical kinetic energy.

    Parameters
    ----------
    velocities : ndarray, shape (N, 3)
        Velocities in a.u.
    masses : ndarray, shape (N,)
        Atomic masses in electron mass units.

    Returns
    -------
    float
    """
    return 0.5 * np.sum(masses[:, None] * velocities ** 2)


# =========================
# Finite difference checks
# =========================

def finite_difference_gradient(
    calculator,
    coords_angstrom,
    symbols,
    delta=1.0e-4,
):
    """
    Numerical gradient via central finite differences.

    Parameters
    ----------
    coords_angstrom : ndarray, shape (N, 3)
        Cartesian coordinates in angstroms.
    delta : float
        Displacement in angstroms.

    Returns
    -------
    grad : ndarray, shape (N, 3)
        Gradient in Hartree / bohr.
    """
    natom = coords_angstrom.shape[0]
    grad = np.zeros_like(coords_angstrom)

    for i in range(natom):
        for j in range(3):
            disp = np.zeros_like(coords_angstrom)
            disp[i, j] = delta

            geom_p = build_psi4_geometry(
                coords_angstrom + disp, symbols, units="angstrom"
            )
            geom_m = build_psi4_geometry(
                coords_angstrom - disp, symbols, units="angstrom"
            )

            Ep = calculator.energy(geom_p)
            Em = calculator.energy(geom_m)

            grad[i, j] = (Ep - Em) / (2 * delta * ANGSTROM_TO_BOHR)

    return grad

def finite_difference_gradient_from_geometry(
    calculator,
    geometry,
    delta=1.0e-4,
):
    """
    FD gradient using a Psi4 geometry string.

    Returns gradient in Hartree / bohr.
    """
    symbols, coords = parse_psi4_geometry(geometry)
    return finite_difference_gradient(
        calculator,
        coords_angstrom=coords,
        symbols=symbols,
        delta=delta,
    )


def write_xyz(
    filename,
    symbols,
    coords_angstrom,
    comment="",
    mode="a",
):
    """
    Write a single XYZ frame.

    Parameters
    ----------
    filename : str
        XYZ file to write to.
    symbols : list[str]
        Atomic symbols.
    coords_angstrom : ndarray, shape (N,3)
        Cartesian coordinates in angstrom.
    comment : str
        Comment line (energy, step, etc.).
    mode : {'w', 'a'}
        Write mode: overwrite or append.
    """
    coords_angstrom = np.asarray(coords_angstrom)
    natom = len(symbols)

    with open(filename, mode) as f:
        f.write(f"{natom}\n")
        f.write(f"{comment}\n")
        for sym, (x, y, z) in zip(symbols, coords_angstrom):
            f.write(f"{sym:2s} {x: .10f} {y: .10f} {z: .10f}\n")

