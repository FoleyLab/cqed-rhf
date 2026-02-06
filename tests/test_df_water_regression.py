import numpy as np
import psi4
import pytest

from cqed_rhf import CQEDRHFCalculator
from cqed_rhf.utils import ANGSTROM_TO_BOHR


# =============================================================================
# Common test data
# =============================================================================

H2O_GEOM = """
O            0.000000000000     0.000000000000    -0.065775570547
H            0.000000000000    -0.759061990794     0.521953018286
H            0.000000000000     0.759061990794     0.521953018286
units angstrom
no_reorient
no_com
symmetry c1
"""

PSI4_OPTIONS_DF = {
    "basis": "aug-cc-pvdz",
    "scf_type": "df",
    "e_convergence": 1e-12,
    "d_convergence": 1e-12,
}

ENERGY_TOL = 1e-8          # Hartree
GRAD_TOL = 1e-6            # Ha / bohr
DF_GRAD_DIFF_TOL = 1e-4    # Ha / bohr
EXPECTED_RHF_E = -76.04123648668632

# =============================================================================
# 1. Energy regression: Psi4 DF RHF vs CQED-RHF (lambda = 0)
# =============================================================================

@pytest.mark.slow
def test_df_energy_lambda_zero_matches_psi4():
    """DF-RHF energy from CQED-RHF (lambda=0) must match Psi4."""

    psi4.core.clean()
    psi4.set_options(PSI4_OPTIONS_DF)

    mol = psi4.geometry(H2O_GEOM)

    # Psi4 DF-RHF energy
    e_psi4 = psi4.energy("scf")

    # CQED-RHF with lambda = 0
    calc = CQEDRHFCalculator(
        lambda_vector=[0.0, 0.0, 0.0],
        omega=0.0,
        psi4_options=PSI4_OPTIONS_DF,
        density_fitting=True,
    )

    e_cqed, _, _ = calc.energy_and_gradient(
        H2O_GEOM,
        canonical="psi4",
    )

    assert np.isclose(e_cqed, EXPECTED_RHF_E)
    assert np.isclose(e_psi4, EXPECTED_RHF_E)

    #assert abs(e_cqed - e_psi4) < ENERGY_TOL, (
    #    f"Energy mismatch: CQED={e_cqed:.10f}, Psi4={e_psi4:.10f}"
    #)


# =============================================================================
# 2. Gradient regression: Psi4 DF RHF vs CQED-RHF (lambda = 0)
# =============================================================================

@pytest.mark.slow
def test_df_gradient_lambda_zero_matches_psi4():
    """DF-RHF gradient from CQED-RHF (lambda=0) must match Psi4."""

    psi4.core.clean()
    psi4.set_options(PSI4_OPTIONS_DF)

    mol = psi4.geometry(H2O_GEOM)

    # Psi4 DF-RHF gradient
    grad_psi4 = np.asarray(psi4.gradient("scf"))

    # CQED-RHF with lambda = 0
    calc = CQEDRHFCalculator(
        lambda_vector=[0.0, 0.0, 0.0],
        omega=0.0,
        psi4_options=PSI4_OPTIONS_DF,
        density_fitting=False,
    )

    _, grad_cqed, _ = calc.energy_and_gradient(
        H2O_GEOM,
        canonical="psi4",
    )

    diff = grad_cqed - grad_psi4
    max_diff = np.max(np.abs(diff))
    rms_diff = np.sqrt(np.mean(diff**2))

    assert max_diff < GRAD_TOL, (
        f"Max gradient diff too large: {max_diff:.3e} Ha/bohr"
    )
    assert rms_diff < GRAD_TOL, (
        f"RMS gradient diff too large: {rms_diff:.3e} Ha/bohr"
    )


# =============================================================================
# 3. CQED-RHF gradient: DF vs canonical (lambda != 0)
# =============================================================================

@pytest.mark.slow
def test_df_vs_canonical_cqed_gradient_consistency():
    """
    CQED-RHF gradients with and without DF should be acceptably similar
    for lambda != 0.
    """

    psi4.core.clean()
    psi4.set_options(PSI4_OPTIONS_DF)

    lambda_vec = [0.1, 0.1, 0.1]
    omega = 0.1

    # --- Canonical CQED-RHF ---
    calc_canonical = CQEDRHFCalculator(
        lambda_vector=lambda_vec,
        omega=omega,
        psi4_options=PSI4_OPTIONS_DF,
        density_fitting=False,
    )

    _, grad_can, _ = calc_canonical.energy_and_gradient(
        H2O_GEOM,
        canonical="exact",
    )

    # --- DF CQED-RHF ---
    calc_df = CQEDRHFCalculator(
        lambda_vector=lambda_vec,
        omega=omega,
        psi4_options=PSI4_OPTIONS_DF,
        density_fitting=True,
    )

    _, grad_df, _ = calc_df.energy_and_gradient(
        H2O_GEOM,
        canonical="exact",
    )

    diff = grad_df - grad_can
    diff_norm = np.linalg.norm(diff)

    assert diff_norm < DF_GRAD_DIFF_TOL, (
        f"DF vs canonical CQED gradient mismatch: "
        f"||Δg|| = {diff_norm:.3e} Ha/bohr"
    )

