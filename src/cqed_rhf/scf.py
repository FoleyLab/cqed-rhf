import psi4
import numpy as np
import time
import opt_einsum as oe


class DIISSubspace:
    def __init__(self, max_dim=8):
        self.max_dim = max_dim
        self.errors = []
        self.focks = []

    def add(self, error, fock):
        self.errors.append(error)
        self.focks.append(fock)
        if len(self.errors) > self.max_dim:
            self.errors.pop(0)
            self.focks.pop(0)

    def extrapolate(self):
        n = len(self.errors)
        B = np.empty((n + 1, n + 1))
        B[-1, :] = -1
        B[:, -1] = -1
        B[-1, -1] = 0

        for i in range(n):
            for j in range(n):
                B[i, j] = np.dot(self.errors[i].ravel(), self.errors[j].ravel())

        rhs = np.zeros(n + 1)
        rhs[-1] = -1

        coeff = np.linalg.solve(B, rhs)[:-1]
        F = sum(c * f for c, f in zip(coeff, self.focks))
        return F


class CQEDRHFSCF:
    def __init__(self, geometry, lambda_vector, psi4_options, omega):
        self.geometry = geometry
        self.lambda_vector = np.asarray(lambda_vector)
        self.psi4_options = psi4_options
        self.omega = omega

    def run(self):
        print("Starting CQED-RHF SCF calculation...")
        psi4.set_options(self.psi4_options)
        mol = psi4.geometry(self.geometry)

        # Reference RHF
        rhf_energy, wfn = psi4.energy("scf", return_wfn=True)
        self.wfn = wfn

        mints = psi4.core.MintsHelper(wfn.basisset())

        ndocc = wfn.nalpha()
        nbf = wfn.nmo()

        C = np.asarray(wfn.Ca())
        Cocc = C[:, :ndocc]
        D = oe.contract("pi,qi->pq", Cocc, Cocc, optimize="optimal")

        T = np.asarray(mints.ao_kinetic())
        V = np.asarray(mints.ao_potential())
        S = np.asarray(mints.ao_overlap())

        mu_ao = np.asarray(mints.ao_dipole())

        mu_nuc = np.array([mol.nuclear_dipole()[0], mol.nuclear_dipole()[1], mol.nuclear_dipole()[2]])
        d_ao = sum(self.lambda_vector[i] * mu_ao[i] for i in range(3))

        # Quadrupole
        Q = [np.asarray(x) for x in mints.ao_quadrupole()]
        Q_PF = (
            -0.5 * self.lambda_vector[0] ** 2 * Q[0]
            -0.5 * self.lambda_vector[1] ** 2 * Q[3]
            -0.5 * self.lambda_vector[2] ** 2 * Q[5]
            -self.lambda_vector[0] * self.lambda_vector[1] * Q[1]
            -self.lambda_vector[0] * self.lambda_vector[2] * Q[2]
            -self.lambda_vector[1] * self.lambda_vector[2] * Q[4]
        )

        H = T + V + Q_PF
        H0 = T + V

        A = psi4.core.Matrix.from_array(S)
        A.power(-0.5, 1e-16)
        A = np.asarray(A)

        Enuc = mol.nuclear_repulsion_energy()

        diis = DIISSubspace(max_dim=8)
        Eold = 0.0
        I = np.asarray(mints.ao_eri())

        for it in range(1, 501):
            
            J = oe.contract("pqrs,rs->pq", I, D, optimize="optimal")
            K = oe.contract("prqs,rs->pq", I, D, optimize="optimal")
            N = oe.contract("pr,qs,rs->pq", d_ao, d_ao, D, optimize="optimal")

            F = H + 2 * J - K - N

            err = F @ D @ S - S @ D @ F
            diis.add(err, F)

            diss_e = A.dot(err).dot(A)
            dRMS = np.mean(diss_e**2) ** 0.5

            E = oe.contract("pq,pq->", F + H, D) + Enuc

            if abs(E - Eold) < self.psi4_options.get("e_convergence", 1e-7) and dRMS < self.psi4_options.get("d_convergence", 1e-7):
                break
            Eold = E

            if it > 2:
                F = diis.extrapolate()

            Fp = A @ F @ A
            eps, C2 = np.linalg.eigh(Fp)
            C = A @ C2
            Cocc = C[:, :ndocc]
            D = oe.contract("pi,qi->pq", Cocc, Cocc, optimize="optimal")

        mu_el = np.array([2 * oe.contract("pq,pq->", mu_ao[i], D) for i in range(3)])
        d_exp_el = sum(self.lambda_vector[i] * mu_el[i] for i in range(3))
        d_nuc = sum(self.lambda_vector[i] * mu_nuc[i] for i in range(3))
        d_exp = d_exp_el + d_nuc

        results = dict(
            energy=E,
            density=D,
            coefficients=C,
            orbital_energies=eps,
            mints=mints,
            wfn=wfn, #<-- Note this is the wfn from converged RHF, not CQED-RHF
            dipole_el=mu_el,
            dipole_nuc=mu_nuc,
            d_ao=d_ao,
            d_exp=d_exp,
            H0=H0,
            F=F,
            ndocc=ndocc,
            natom=mol.natom(),
        )

        return E, results

