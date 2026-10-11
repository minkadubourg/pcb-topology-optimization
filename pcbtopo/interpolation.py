"""Material interpolation and density filtering."""
from __future__ import annotations

import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree


class SIMP:
    """sigma(rho) = sigma_min + (sigma_max - sigma_min) * rho**p.

    ``p = 1`` is the linear law of the original study; ``p = 3`` penalises
    intermediate densities, which pushes designs toward copper / no copper
    without an explicit grey-material constraint.
    """

    def __init__(self, sigma_min: float = 1e-6, sigma_max: float = 1.0, p: float = 3.0):
        self.sigma_min, self.sigma_max, self.p = sigma_min, sigma_max, p

    def __call__(self, rho):
        return self.sigma_min + (self.sigma_max - self.sigma_min) * np.asarray(rho) ** self.p

    def derivative(self, rho):
        rho = np.asarray(rho)
        return (self.sigma_max - self.sigma_min) * self.p * np.maximum(rho, 0) ** (self.p - 1)


class DensityFilter:
    """Linear density filter (Bruns & Tortorelli 2001; Bourdin 2001):
    rho_tilde_e = sum_j w_ej A_j rho_j / sum_j w_ej A_j,  w_ej = max(0, r - |c_e - c_j|).

    Removes checkerboards and sets a minimum feature size of about 2r.
    ``radius = 0`` is the identity.
    """

    def __init__(self, centroids: np.ndarray, areas: np.ndarray, radius: float):
        n = len(centroids)
        if radius <= 0:
            self.W = sp.identity(n, format="csr")
            return
        tree = cKDTree(centroids)
        pairs = tree.query_pairs(radius, output_type="ndarray")
        i = np.concatenate([pairs[:, 0], pairs[:, 1], np.arange(n)])
        j = np.concatenate([pairs[:, 1], pairs[:, 0], np.arange(n)])
        w = (radius - np.linalg.norm(centroids[i] - centroids[j], axis=1)) * areas[j]
        H = sp.csr_matrix((w, (i, j)), shape=(n, n))
        self.W = sp.diags(1.0 / np.asarray(H.sum(axis=1)).ravel()) @ H

    def __call__(self, rho):
        return self.W @ rho

    def backward(self, grad_filtered):
        """Chain rule: gradient with respect to the unfiltered densities."""
        return self.W.T @ grad_filtered


def element_centroids(mesh) -> np.ndarray:
    from ngsolve import VOL

    pts = np.array([v.point for v in mesh.vertices])
    return np.array([pts[[v.nr for v in el.vertices]].mean(axis=0) for el in mesh.Elements(VOL)])
