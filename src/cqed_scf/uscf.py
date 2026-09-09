"""Unrestricted CQED-SCF engine (UHF / UKS references).

The restricted engine in :mod:`cqed_scf.scf` handles closed-shell RHF and RKS.
This module is its open-shell counterpart, reached automatically by
:class:`~cqed_scf.calculator.CQEDCalculator` whenever the configured reference
is ``"uhf"`` or ``"uks"``.

The surrounding plumbing is complete: :class:`CQEDConfig` accepts and validates
unrestricted references, the calculator checks the geometry's charge and
multiplicity against the config and dispatches here, and gradients and response
theory refuse unrestricted references up front.  What is missing is the physics
below.

Reference energies to implement against live in
``examples/canonical/Unrestricted_Test_Examples/`` (Psi4 for UHF/UKS, Hilbert
``polaritonic_scf`` for QED-UHF/QED-UKS, cross-validated to <5e-13 Eh at
lambda = 0).

Implementation notes
--------------------
Start from :class:`cqed_scf.scf.CQEDSCF` and split each spin channel:

* Read the functional from ``config.base_scf_functional``, **not**
  ``config.functional``.  The former returns ``None`` for ``"uhf"`` and strips
  known dispersion suffixes; the latter would try to build a superfunctional for
  a Hartree-Fock reference.
* The restricted code carries an explicit factor of 2 throughout because its
  ``D`` is the alpha density alone.  With separate ``Da``/``Db`` those factors
  disappear and ``Dt = Da + Db`` takes their place -- in the Coulomb term, in the
  dipole expectation value at ``scf.py:341``, and in the dipole self-energy.
* :meth:`~cqed_scf.scf.CQEDSCF._build_JK` adds a single ``C_left_add(Cocc)``,
  relying on ``J_alpha == J_beta``.  Unrestricted needs both occupied blocks
  pushed through, with ``J`` summed over spins and ``K`` kept spin-resolved.
* :meth:`~cqed_scf.scf.CQEDSCF._build_vbase` builds a ``"RV"`` potential.  UKS
  needs ``"UV"``, with ``set_D([Da, Db])`` and ``compute_V([Va, Vb])``.
* The DIIS subspace in ``scf.py`` stores flat error/Fock arrays.  Either keep two
  subspaces or stack the alpha and beta errors into one vector.

Return contract
---------------
:meth:`run` must return ``(energy, results)``.  ``CQEDCalculator._run_scf``
returns that tuple unchanged, and :meth:`CQEDCalculator.energy` immediately reads
``results["energy_psi4"]``, so that key is mandatory.

``results`` should carry the keys the restricted engine produces (see the dict
built at the end of ``cqed_scf/scf.py``) so that shared consumers keep working,
with these spin-resolved additions and substitutions:

===========================  ==================================================
key                          meaning
===========================  ==================================================
``energy_scf``               total CQED-SCF energy
``energy_psi4``              the underlying Psi4 UHF/UKS energy (**required**)
``Ca``, ``Cb``               alpha/beta MO coefficients
``Da``, ``Db``               alpha/beta AO densities (each *unscaled*)
``orbital_energies_a/_b``    alpha/beta orbital energies
``nalpha``, ``nbeta``        occupied counts per spin
``s_squared``                <S^2> for the converged determinant
===========================  ==================================================

Keep ``density``, ``coefficients``, ``orbital_energies``, and ``ndocc`` out of
the unrestricted dict rather than aliasing them to the alpha channel.  Several
consumers (``cqed_scf.gradients``, ``cqed_scf.response``) read those keys and
silently assume the restricted factor-of-2 convention; a missing key raises,
while a plausible-looking alias would produce a wrong number.
"""

from __future__ import annotations

from typing import Any

from .references import CQEDConfig


class CQEDUSCF:
    """Unrestricted CQED-SCF driver.

    Intended workflow:
    1. Build separate alpha/beta reference wavefunctions from Psi4 UHF/UKS.
    2. Construct CQED one-electron and dipole self-energy contributions.
    3. Iterate alpha/beta Fock builds to convergence.
    4. Return ``(energy, results)`` following the contract in the module
       docstring.
    """

    def __init__(self, geometry: Any, config: CQEDConfig):
        self.geometry = geometry
        self.config = config

    def run(self):
        """Run unrestricted CQED-SCF.

        Returns
        -------
        tuple[float, dict]
            The CQED-SCF energy and the results dictionary described in the
            module docstring.
        """

        raise NotImplementedError(
            "Unrestricted CQED-SCF physics is not implemented yet. The "
            f"reference {self.config.reference!r} and multiplicity "
            f"{self.config.multiplicity} were validated and dispatched here "
            "correctly; what remains is the alpha/beta SCF itself. See the "
            "cqed_scf.uscf module docstring for the return contract and "
            "examples/canonical/Unrestricted_Test_Examples/ for reference "
            "energies."
        )


CQEDUHFSCF = CQEDUSCF
