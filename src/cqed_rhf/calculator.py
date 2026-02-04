import psi4
from .scf import CQEDRHFSCF
from .gradients import CQEDRHFGradient


class CQEDRHFCalculator:
    def __init__(self, lambda_vector, psi4_options, omega=0.1):
        self.lambda_vector = lambda_vector
        self.psi4_options = psi4_options
        self.omega = omega

    def energy(self, geometry):
        scf = CQEDRHFSCF(
            geometry, self.lambda_vector, self.psi4_options, self.omega
        )
        E, _ = scf.run()
        psi4.core.clean()
        return E

    def energy_and_gradient(self, geometry, canonical="psi4"):
        scf = CQEDRHFSCF(
            geometry, self.lambda_vector, self.psi4_options, self.omega
        )
        E, data = scf.run()

        grad_engine = CQEDRHFGradient(
            self.lambda_vector, canonical=canonical
        )
        grad = grad_engine.compute(data)

        g = (self.omega / 2) ** 0.5 * data["d_exp"]

        psi4.core.clean()
        psi4.core.clean_options()
        return E, grad, g


