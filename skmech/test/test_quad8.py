"""test quad 8 node element"""
import numpy as np
import sympy as sp
from skmech.elements.quad8 import Quad8


def shapefunc(xiv, etav):
    """Test shape functions"""

    Xi = np.array([[-1, -1], [1, -1], [1, 1], [-1, 1], [0, -1], [1, 0], [0, 1],
                   [-1, 0]])
    xi_i, eta_i = Xi[:, 0], Xi[:, 1]
    xi, eta = sp.symbols('xi, eta')
    N = np.array([
        1 / 4 * (1 + xi_i[0] * xi) * (1 + eta_i[0] * eta),
        1 / 4 * (1 + xi_i[1] * xi) * (1 + eta_i[1] * eta),
        1 / 4 * (1 + xi_i[2] * xi) * (1 + eta_i[2] * eta),
        1 / 4 * (1 + xi_i[3] * xi) * (1 + eta_i[3] * eta),
        1 / 2 * (1 - xi**2) * (1 + eta * eta_i[4]),
        1 / 2 * (1 - eta**2) * (1 + xi * xi_i[5]),
        1 / 2 * (1 - xi**2) * (1 + eta * eta_i[6]),
        1 / 2 * (1 - eta**2) * (1 + xi * xi_i[7])
    ])
    dN_Xi = []
    for nf in N:
        dN_Xi.append([nf.diff(xi), nf.diff(eta)])
    dN_Xi = np.array(dN_Xi)

    vN, vdN_Xi = [], []
    for Nj, dN_Xij in zip(N, dN_Xi):
        vN.append(Nj.subs({xi: xiv, eta: etav}))
        vdN_Xi.append([dN_Xij[0].subs({xi: xiv, eta: etav}),
                       dN_Xij[1].subs({xi: xiv, eta: etav})])
    return np.array(vN), np.array(vdN_Xi).T


aN, adN_Xi = shapefunc(-1, -1)
print(aN)
print(adN_Xi)
