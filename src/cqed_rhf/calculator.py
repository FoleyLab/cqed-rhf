import psi4
from .scf import CQEDRHFSCF
from .gradients import CQEDRHFGradient
import time


class CQEDRHFCalculator:
    def __init__(self, lambda_vector, psi4_options, omega=0.1, density_fitting=False, debug=False):
        self.lambda_vector = lambda_vector
        self.psi4_options = psi4_options
        self.omega = omega
        self.density_fitting = density_fitting
        self.debug = debug
        

    def energy(self, geometry):
        scf = CQEDRHFSCF(
            geometry, self.lambda_vector, self.psi4_options, self.omega, self.density_fitting, self.debug
        )
        E, _ = scf.run()
        psi4.core.clean()
        return E

    def energy_and_gradient(self, geometry, canonical="psi4"):
        t0 = time.time()
        scf = CQEDRHFSCF(
            geometry, self.lambda_vector, self.psi4_options, self.omega, self.density_fitting, self.debug
        )
        print("Instantiating SCF time: {:.4f} s".format(time.time() - t0))
        t0 = time.time()
        E, data = scf.run()
        print("SCF run time: {:.4f} s".format(time.time() - t0))

        t0 = time.time()
        grad_engine = CQEDRHFGradient(
            self.lambda_vector, canonical=canonical, debug=self.debug
        )
        print("Instantiating gradient engine time: {:.4f} s".format(time.time() - t0))
        t0 = time.time()
        grad = grad_engine.compute(data)
        print("Gradient computation time: {:.4f} s".format(time.time() - t0))
        g = (self.omega / 2) ** 0.5 * data["d_exp"]

        psi4.core.clean()
        psi4.core.clean_options()
        return E, grad,  g


