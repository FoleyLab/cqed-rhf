import numpy as np


class RotationalProjectionObserver:
    """
    Projects the full nuclear gradient onto rigid-body rotational
    modes corresponding to phi and theta orientation angles.
    """

    def __init__(self, orientation_tracker, masses):
        """
        Parameters
        ----------
        orientation_tracker : NitrobenzeneOrientation
            Provides x_hat, z_hat, field_hat.
        masses : ndarray, shape (N,)
            Atomic masses in atomic units.
        """
        self.orientation_tracker = orientation_tracker
        self.masses = masses

    def _center_of_mass(self, coords):
        total_mass = np.sum(self.masses)
        return np.sum(coords * self.masses[:, None], axis=0) / total_mass

    def _rotation_axis(self, v_hat, field_hat):
        axis = np.cross(v_hat, field_hat)
        norm = np.linalg.norm(axis)
        if norm < 1e-12:
            return None  # aligned, no defined rotation axis
        return axis / norm

    def observe(self, coords_bohr, grad):
        """
        Parameters
        ----------
        coords_bohr : ndarray (N,3)
        grad : ndarray (N,3)

        Returns
        -------
        dict
        """

        # --- Get orientation vectors ---
        x_hat, z_hat = self.orientation_tracker._compute_vectors(coords_bohr)
        field_hat = self.orientation_tracker.field_hat

        # --- Remove translation (COM frame) ---
        com = self._center_of_mass(coords_bohr)
        R = coords_bohr - com

        # --- Rotation axes ---
        axis_phi = self._rotation_axis(x_hat, field_hat)
        axis_theta = self._rotation_axis(z_hat, field_hat)

        results = {}

        # --- Compute torque vector (optional but useful) ---
        forces = -grad
        torque = np.sum(np.cross(R, forces), axis=0)

        results["torque_vector"] = torque

        # --- dE/dphi ---
        if axis_phi is not None:
            dR_dphi = np.cross(axis_phi, R)
            dE_dphi = np.sum(grad * dR_dphi)
            results["axis_phi"] = axis_phi
            results["dE_dphi"] = dE_dphi
            results["torque_phi"] = -np.dot(torque, axis_phi)
        else:
            results["axis_phi"] = None
            results["dE_dphi"] = 0.0
            results["torque_phi"] = 0.0

        # --- dE/dtheta ---
        if axis_theta is not None:
            dR_dtheta = np.cross(axis_theta, R)
            dE_dtheta = np.sum(grad * dR_dtheta)
            results["axis_theta"] = axis_theta
            results["dE_dtheta"] = dE_dtheta
            results["torque_theta"] = -np.dot(torque, axis_theta)
        else:
            results["axis_theta"] = None
            results["dE_dtheta"] = 0.0
            results["torque_theta"] = 0.0

        return results

