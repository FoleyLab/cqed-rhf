import numpy as np

class NitrobenzeneOrientation:
    def __init__(self, symbols, coords):
        """
        symbols : list[str]
        coords  : (N,3) ndarray in BOHR
        """
        self.symbols = symbols
        self.C_indices = [i for i, s in enumerate(symbols) if s == "C"]
        self.N_index   = [i for i, s in enumerate(symbols) if s == "N"][0]

        # identify bonded carbon ONCE
        self.CN_index = self._find_bonded_carbon(coords)

        # choose ring reference atoms ONCE
        self.ring_ref = self._choose_ring_triplet()

    def _find_bonded_carbon(self, coords):
        N = coords[self.N_index]
        C_coords = coords[self.C_indices]
        dists = np.linalg.norm(C_coords - N, axis=1)
        return self.C_indices[np.argmin(dists)]

    def _choose_ring_triplet(self):
        """
        Pick 3 non-collinear carbons in the ring.
        For benzene, any 3 works as long as they aren't collinear.
        """
        return self.C_indices[:3]

    def compute_vectors(self, coords):
        """
        Returns x_hat, z_hat
        """
        # ---- x_hat: C-N bond
        x_vec = coords[self.N_index] - coords[self.CN_index]
        x_hat = x_vec / np.linalg.norm(x_vec)

        # ---- z_hat: ring normal
        i, j, k = self.ring_ref
        v1 = coords[j] - coords[i]
        v2 = coords[k] - coords[i]
        z_vec = np.cross(v1, v2)
        z_hat = z_vec / np.linalg.norm(z_vec)

        # enforce sign convention (optional)
        if z_hat[2] < 0:
            z_hat = -z_hat

        return x_hat, z_hat

    def align_with_field(self, coords, field_vec):
        """
        Returns cos(phi), cos(theta)
        """
        field_hat = field_vec / np.linalg.norm(field_vec)
        x_hat, z_hat = self.compute_vectors(coords)

        cos_phi   = np.dot(x_hat, field_hat)
        cos_theta = np.dot(z_hat, field_hat)
        phi = np.acos(cos_phi)
        theta = np.acos(cos_theta)

        return phi, theta

