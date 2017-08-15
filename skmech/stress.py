"""Compute the stress for each node

"""
import numpy as np
from .constructor import constructor


def recovery(model, U, EPS0, t=1):
    """Recovery stress at nodes from displacement

    """
    # initiate the arrays for element and global stress
    sig = np.zeros(3)
    SIG = np.zeros((model.num_nodes, 3))

    for e, conn in enumerate(model.CONN):
        element = constructor(e, model, EPS0)
        dof = element.dof
        xyz = element.xyz

        # if there is no initial strain
        if EPS0 is None:
            eps0 = np.zeros(3)
        else:
            eps0 = EPS0[e]

        u = U[dof]

        # quadrature on the nodes coord in the isodomain
        for n, gp in enumerate(element.xez):
            _, dN_ei = element.shape_function(gp)
            dJ, dN_xi, _ = element.jacobian(xyz, dN_ei)

            # number of elements sharing a node
            num_ele_shrg = (model.CONN == conn[n]).sum()

            if callable(element.E):
                x1, x2 = element.mapping(element.xyz)
                C = element.c_matrix(t, x1, x2)
            else:
                C = element.c_matrix(t)

            B = np.array([
               [dN_xi[0, 0], 0, dN_xi[0, 1], 0, dN_xi[0, 2], 0,
                dN_xi[0, 3], 0],
               [0, dN_xi[1, 0], 0, dN_xi[1, 1], 0, dN_xi[1, 2], 0,
                dN_xi[1, 3]],
               [dN_xi[1, 0], dN_xi[0, 0], dN_xi[1, 1], dN_xi[0, 1],
                dN_xi[1, 2], dN_xi[0, 2], dN_xi[1, 3], dN_xi[0, 3]]])

            # sig = [sig_11 sig_22 sig_12] for each n node
            sig = C @ (B @ u - eps0)

            # dof 1 degree of freedom per node
            d = int(dof[2*n]/2)

            # unweighted average of stress at nodes
            SIG[d, :] += sig/num_ele_shrg

    return SIG


def principal_max(s11, s22, s12):
    """Compute the principal stress max

    """
    sp_max = np.zeros(len(s11))
    for i in range(len(s11)):
        sp_max[i] = (s11[i] + s22[i]) / 2.0 + np.sqrt(
            (s11[i] - s22[i])**2.0 / 2.0 + s12[i]**2.0)
    return sp_max


def principal_min(s11, s22, s12):
    """Compute the principal stress minimum

    """
    sp_min = np.zeros(len(s11))
    for i in range(len(s11)):
        sp_min[i] = (s11[i]+s22[i])/2. - np.sqrt((s11[i] - s22[i])**2./2. +
                                                 s12[i]**2.)
    return sp_min


def recovery_gauss(model, U, EPS0, t=1):
    """Recovery stress at gauss points from displacement

    """
    # initiate the arrays for element and global stress
    SIG = np.zeros((model.num_nodes, 3))
    SIG2 = np.zeros((model.num_nodes, 3))
    # extrapolation matrix
    Q = np.array([[1 + np.sqrt(3)/2, -1/2, 1 - np.sqrt(3)/2, -1/2],
                  [-1/2, 1 + np.sqrt(3)/2, -1/2, 1 - np.sqrt(3)/2],
                  [1 - np.sqrt(3)/2, -1/2, 1 + np.sqrt(3)/2, -1/2],
                  [-1/2, 1 - np.sqrt(3)/2, -1/2, 1 + np.sqrt(3)/2]])

    for e, conn in enumerate(model.CONN):
        element = constructor(e, model, EPS0)
        dof = element.dof
        xyz = element.xyz

        # if there is no initial strain
        if EPS0 is None:
            eps0 = np.zeros(3)
        else:
            eps0 = EPS0[e]

        u = U[dof]

        # quadrature on the nodes coord in the isodomain
        for n, gp in enumerate(element.xez/np.sqrt(3)):
            N, dN_ei = element.shape_function(gp)
            dJ, dN_xi, _ = element.jacobian(xyz, dN_ei)

            C = element.c_matrix(N, t)

            # number of elements sharing a node
            num_ele_shrg = (model.CONN == conn[n]).sum()

            B = np.array([
               [dN_xi[0, 0], 0, dN_xi[0, 1], 0, dN_xi[0, 2], 0,
                dN_xi[0, 3], 0],
               [0, dN_xi[1, 0], 0, dN_xi[1, 1], 0, dN_xi[1, 2], 0,
                dN_xi[1, 3]],
               [dN_xi[1, 0], dN_xi[0, 0], dN_xi[1, 1], dN_xi[0, 1],
                dN_xi[1, 2], dN_xi[0, 2], dN_xi[1, 3], dN_xi[0, 3]]])

            # sig = [sig_11 sig_22 sig_12] for each n node
            sig_gp = C @ (B @ u - eps0)

            # 1 degree of freedom per node
            # unweighted average of stress at nodes
            SIG[conn[n], :] += sig_gp/num_ele_shrg

        SIG2[conn, 0] = Q @ SIG[conn, 0]
        SIG2[conn, 1] = Q @ SIG[conn, 1]
        SIG2[conn, 2] = Q @ SIG[conn, 2]

    return SIG2


def recovery_at_gp(U, model, t=1):
    """recovery stresses at gauss points (gp)

    Args:
        U (numpy array shape(num_dof, 1)): results for each degree of freedom
        model (obj): object with model attributes
        t (float): time

    Returns:
        SIG (numpy array shape(num_ele*num_gp, 5)) with the coordinates of each
            gauss points coordinates in the cartesian system, and the stress
            values at those points. Example
                [[gp1_x, gp2_y, s11, s22, s13], ...]
    """
    sig = []
    for e, conn in enumerate(model.CONN):
        element = constructor(e, model)

        u = U[element.dof]

        # loop over quadrature points
        for w, gp in zip(element.gauss_quad.weights,
                         element.gauss_quad.points):

            N, dN_ei = element.shape_function(xez=gp)
            dJ, dN_xi, _ = element.jacobian(element.xyz, dN_ei)

            C = element.c_matrix(N, t)

            # Standard geadient operator matrix (stain-displacement)
            B_s = []
            for j in range(element.num_std_nodes):
                B_s.append(np.array([[dN_xi[0, j], 0],
                                     [0, dN_xi[1, j]],
                                     [dN_xi[1, j], dN_xi[0, j]]]))
            Bstd = np.block([B_s[i] for i in range(element.num_std_nodes)])

            if model.xfem:

                # loop for each zero level set
                Benr_zls = {}   # Benr for each zerp level est
                for ind, zls in enumerate(element.zerolevelset):
                    # signed distance for nodes in this element for this zls
                    phi = zls.phi[element.conn]  # phi with local index
                    # Enriched gradient operator matrix
                    B_e = {}
                    for n in element.enriched_nodes[ind]:
                        # local reference of node n in element
                        j = element.global2local_index(n)
                        psi = abs(N @ phi) - abs(phi[j])

                        dpsi_x = np.sign(N @ phi)*(dN_xi[0, :] @ phi)
                        dpsi_y = np.sign(N @ phi)*(dN_xi[1, :] @ phi)
                        B_e[n] = np.array([
                            [dN_xi[0, j]*(psi) + N[j]*dpsi_x, 0],
                            [0, dN_xi[1, j]*(psi) + N[j]*dpsi_y],
                            [dN_xi[1, j]*(psi) + N[j]*dpsi_y,
                             dN_xi[0, j]*(psi) + N[j]*dpsi_x]])

                    Benr_zls[ind] = np.block([B_e[i]
                                              for i
                                              in element.enriched_nodes[ind]])

                Benr = np.block([Benr_zls[i]
                                 for i in range(len(element.zerolevelset))])

                B = np.block([Bstd, Benr])
            else:
                B = Bstd

            # TODO: add initial strain due thermal changes
            s = C @ (B @ u)
            x, y = element.mapping(N, element.xyz)
            sig.append([x, y, s[0], s[1], s[2]])

    return np.array(sig)
