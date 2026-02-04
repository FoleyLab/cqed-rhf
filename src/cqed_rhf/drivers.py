import psi4
import numpy as np
from scipy.optimize import minimize

from .utils import (
    build_psi4_geometry,
    parse_psi4_geometry,
    write_xyz,
    ANGSTROM_TO_BOHR,
    AMU_TO_AU,
)

def initialize_md_from_geometry(geometry_string):
    """
    Initialize MD coordinates, symbols, and masses from a Psi4 geometry string.

    Returns
    -------
    coords_bohr : ndarray, shape (N,3)
    symbols : list[str]
    masses : ndarray, shape (N,)
        Atomic masses in electron mass units.
    mol : psi4.core.Molecule
    """
    mol = psi4.geometry(geometry_string)

    natom = mol.natom()
    symbols = [mol.symbol(i) for i in range(natom)]
    masses = np.array([mol.mass(i) for i in range(natom)]) * AMU_TO_AU
    coords_bohr = mol.geometry().to_array()

    return coords_bohr, symbols, masses, mol


def velocity_verlet_md(
    calculator,
    geometry=None,
    coords=None,
    symbols=None,
    velocities=None,
    dt=10.0,
    nsteps=10,
    canonical="psi4",
    debug=False,
):
    """
    Velocity-Verlet molecular dynamics.

    Parameters
    ----------
    calculator : CQEDRHFCalculator
    geometry : str, optional
        Psi4 geometry string (angstroms).
    coords : ndarray, shape (N,3), optional
        Cartesian coordinates in angstroms.
    symbols : list[str]
        Atomic symbols.
    velocities : ndarray, shape (N,3), optional
        Initial velocities (bohr / a.u. time).
        Defaults to zero.
    dt : float
        Time step in atomic units.
    nsteps : int
        Number of MD steps.
    canonical : {'psi4', 'exact'}
        Gradient backend.
    debug : bool
        Print energies each step.

    Returns
    -------
    traj : list of dict
        Trajectory data.
    """

    # -------------------------
    # Initialize system
    # -------------------------
    if geometry is not None:
        # Psi4 geometry string is the source of truth
        coords_bohr, symbols, masses, mol = initialize_md_from_geometry(geometry)

    elif coords is not None and symbols is not None:
        # User provided coords in angstroms
        coords = np.asarray(coords)
        coords_bohr = coords * ANGSTROM_TO_BOHR

        # Build temporary molecule to get masses
        geom = build_psi4_geometry(coords, symbols, units="angstrom")
        mol = psi4.geometry(geom)

        masses = np.array([mol.mass(i) for i in range(mol.natom())]) * AMU_TO_AU

    else:
        raise ValueError("Provide either geometry or coords+symbols.")

    natom = len(symbols)

    # -------------------------
    # Velocities (bohr / a.u.)
    # -------------------------
    if velocities is None:
        velocities = np.zeros((natom, 3))
    else:
        velocities = np.asarray(velocities)

    # coords_bohr is now the authoritative MD position array

    # -------------------------
    # Initial forces
    # -------------------------
    coords_angstrom = coords_bohr / ANGSTROM_TO_BOHR
    geom = build_psi4_geometry(coords_angstrom, symbols, units="angstrom")
    E, grad, g = calculator.energy_and_gradient(
        geom, canonical=canonical
    )

    forces = -grad  # Hartree / bohr

    traj = []

    # -------------------------
    # MD loop
    # -------------------------
    for step in range(nsteps):

        # Half-step velocity update
        velocities -= 0.5 * dt * forces / masses[:, None]

        # Position update (bohr)
        coords_bohr += dt * velocities

        # Back to angstroms for Psi4
        coords_angstrom = coords_bohr / ANGSTROM_TO_BOHR
        geom = build_psi4_geometry(coords_angstrom, symbols, units="angstrom")

        # New forces
        E, grad, g = calculator.energy_and_gradient(
            geom, canonical=canonical
        )
        forces = -grad

        # Final half-step velocity update
        velocities -= 0.5 * dt * forces / masses[:, None]

        # Store step
        traj.append(
            dict(
                step=step,
                energy=E,
                coords=coords.copy(),
                velocities=velocities.copy(),
                forces=forces.copy(),
                coupling=g,
            )
        )

        if debug:
            print(
                f"Step {step:4d} | "
                f"E = {E: .8f} Ha | "
                f"|F| = {np.linalg.norm(forces):.4e}"
            )

    return traj

def bfgs_optimize(
    calculator,
    geometry,
    canonical="psi4",
    gtol=1e-5,
    maxiter=50,
    debug=False,
):
    """
    Geometry optimization using BFGS.

    Parameters
    ----------
    calculator : CQEDRHFCalculator
    geometry : str
        Psi4 geometry string (angstrom).
    canonical : {'psi4', 'exact'}
        Gradient backend.
    gtol : float
        Gradient norm tolerance (Ha/bohr).
    maxiter : int
        Maximum iterations.
    debug : bool
        Print progress.

    Returns
    -------
    result : OptimizeResult
        SciPy optimization result.
    """

    # Parse initial geometry
    mol = psi4.geometry(geometry)
    symbols = [mol.symbol(i) for i in range(mol.natom())]
    x0_bohr = mol.geometry().to_array()

    def objective(x_flat):
        coords_bohr = x_flat.reshape(-1, 3)
        coords_angstrom = coords_bohr / ANGSTROM_TO_BOHR

        geom = build_psi4_geometry(
            coords_angstrom, symbols, units="angstrom"
        )

        E, grad, g = calculator.energy_and_gradient(
            geom, canonical=canonical
        )

        if debug:
            write_xyz(
                "opt_traj.xyz",
                symbols,
                coords_angstrom,
                comment=f"E={E:.10f} |grad|={np.linalg.norm(grad):.3e}",
                mode="a",
            )

        return E, grad.reshape(-1)

    result = minimize(
        fun=lambda x: objective(x)[0],
        x0=x0_bohr.reshape(-1),
        jac=lambda x: objective(x)[1],
        method="BFGS",
        options=dict(
            gtol=gtol,
            maxiter=maxiter,
            disp=debug,
        ),
    )

    return result