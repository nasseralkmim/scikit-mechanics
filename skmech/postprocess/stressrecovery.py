"""Compute the stress for each node

"""
import numpy as np
from ..constructor import constructor


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
            d = int(dof[2 * n] / 2)

            # unweighted average of stress at nodes
            SIG[d, :] += sig / num_ele_shrg

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
        for w, gp in zip(element.gauss.weights,
                         element.gauss.points):

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


def stress_recovery(model, t=1):
    """recovery stresses at gauss points (gp)

    Parameters
    ----------
    model : Model object
        object with model attributes
    t : float, default 1
    time

    Returns
    -------
    ndarray, shape(num_ele*num_gp, 5))
        with the coordinates of each gauss points coordinates in the cartesian
        system, and the stress values at those points. Example:
            [[gp1_x, gp2_y, s11, s22, s13], ...]

    """
    sig = []
    for eid, [etype, *edata] in model.elements.items():
        element = constructor(eid, etype, model)
        dof = np.asarray(element.dof) - 1  # go to 0 based
        u = model.dof_displacement[dof]

        # loop over quadrature points
        for w, gp in zip(element.gauss.weights,
                         element.gauss.points):

            N, dN_ei = element.shape_function(xez=gp)
            dJ, dN_xi, _ = element.jacobian(element.xyz, dN_ei)

            C = element.c_matrix(N, t)

            # Standard geadient operator matrix (stain-displacement)
            Bstd = element.gradient_operator(dN_xi)

            if model.xfem:
                if eid in model.xfem.enr_elements:
                    Benr = element.enriched_gradient_operator(N, dN_xi)
                    B = np.block([Bstd, Benr])
                else:
                    B = Bstd
            else:
                B = Bstd

            # TODO: add initial strain due thermal changes
            s = C @ (B @ u)
            x, y = element.mapping(N, element.xyz)
            sig.append([x, y, s[0], s[1], s[2]])

    return np.array(sig)


def stress_recovery_smoothed(model, t=1):
    """Recovery stress at gauss point then extracpolate to nodes

    The stress is computed at the gauss points then extrapolated to element
    nodes using the same shape functions. The extrapolation is also multiplied
    by a weight that takes into account the number of elements sharing a node.

    Parameters
    ----------
    model : Model object
        object with model attributes
    t : float, default 1
        time

    Returns
    -------
    dict
        {node id: [sx, sy, txy]} smoothed stresses

    """
    sig = {}
    for eid, [etype, *edata] in model.elements.items():
        element = constructor(eid, etype, model)
        dof = np.asarray(element.dof) - 1  # go to 0 based
        u = model.dof_displacement[dof]

        # to store 4 values at gauss points
        sig_x_gp, sig_y_gp, sig_xy_gp = [], [], []

        # to form a square from which the interpolation to nodes will occur
        ges = np.max(element.gauss.points)  # gauss element size
        point_to_extrapolate = np.array([[-1 / ges, -1 / ges],
                                         [1 / ges, -1 / ges],
                                         [1 / ges, 1 / ges],
                                         [-1 / ges, 1 / ges]])
        Q = matrix_gp2node(pte=point_to_extrapolate)

        # obtain stresses at GP
        # loop over quadrature points
        for gp in point_to_extrapolate:

            N, dN_ei = element.shape_function(xez=gp)
            dJ, dN_xi, _ = element.jacobian(element.xyz, dN_ei)

            C = element.c_matrix(N, t)

            # Standard geadient operator matrix (stain-displacement)
            Bstd = element.gradient_operator(dN_xi)

            if model.xfem:
                if eid in model.xfem.enr_elements:
                    Benr = element.enriched_gradient_operator(N, dN_xi)
                    B = np.block([Bstd, Benr])
                else:
                    B = Bstd
            else:
                B = Bstd

            # TODO: add initial strain due thermal changes
            # stress at quadrature point s is shape(, 3)
            sx, sy, sxy = C @ (B @ u)

            # append the GP stress values into a list len(4)
            sig_x_gp.append(sx)
            sig_y_gp.append(sy)
            sig_xy_gp.append(sxy)

        # extrapolate from GP to nodes
        sig_x_node = Q @ sig_x_gp
        sig_y_node = Q @ sig_y_gp
        sig_xy_node = Q @ sig_xy_gp

        # create an empty list for the nodal stresses to collect from different
        # elements
        if sig.get(element.conn[0]) is None:
            sig[element.conn[0]] = []
        if sig.get(element.conn[1]) is None:
            sig[element.conn[1]] = []
        if sig.get(element.conn[2]) is None:
            sig[element.conn[2]] = []
        if sig.get(element.conn[3]) is None:
            sig[element.conn[3]] = []

        # append to each node key a vector with the stresses extrapolated
        sig[element.conn[0]].append(np.array([sig_x_node[0],
                                              sig_y_node[0],
                                              sig_xy_node[0]]))
        sig[element.conn[1]].append(np.array([sig_x_node[1],
                                              sig_y_node[1],
                                              sig_xy_node[1]]))
        sig[element.conn[2]].append(np.array([sig_x_node[2],
                                              sig_y_node[2],
                                              sig_xy_node[2]]))
        sig[element.conn[3]].append(np.array([sig_x_node[3],
                                              sig_y_node[3],
                                              sig_xy_node[3]]))

    # smooth the stresses
    # loop over nodes
    for node, stresses in sig.items():
        # each stresses = [ array(sx, sy, sxy), array(sx, sy, sxy)]
        # each array is an extrapolation from an element that share the node
        # len(stresses) means number of elements sharing this nodeq
        if len(stresses) > 1:
            # sum contribution of each element
            sx = sum([s[0] for s in stresses]) / len(stresses)
            sy = sum([s[1] for s in stresses]) / len(stresses)
            sxy = sum([s[2] for s in stresses]) / len(stresses)

            # change the list for an array
            sig[node] = np.array([sx, sy, sxy])
        else:
            # change the list for the unique value in it
            sig[node] = stresses[0]

    return sig


def matrix_gp2node(pte):
    """Construct the matrix for extrapolating from gp to nodes

    ges := gauss_element_size, defines the square gauss element.
    pte := point to extrapoate, considering a coordinate system centered in the
        gauss element, the nodal coordinate is this system is going to be ges.

    Parameters
    ----------
    gauss_element_size : float
        the length of the square where the gauss point values are and from
        where the values will be extrapolated to nodes

    Returns
    -------
    ndarray shape(4, 4)
        matrix that will extrapolate gauss point values into nodes

    """
    Q = np.array([[1 / 4 * (1 - pte[0, 0]) * (1 - pte[0, 1]),
                   1 / 4 * (1 + pte[0, 0]) * (1 - pte[0, 1]),
                   1 / 4 * (1 + pte[0, 0]) * (1 + pte[0, 1]),
                   1 / 4 * (1 - pte[0, 0]) * (1 + pte[0, 1])],
                  [1 / 4 * (1 - pte[1, 0]) * (1 - pte[1, 1]),
                   1 / 4 * (1 + pte[1, 0]) * (1 - pte[1, 1]),
                   1 / 4 * (1 + pte[1, 0]) * (1 + pte[1, 1]),
                   1 / 4 * (1 - pte[1, 0]) * (1 + pte[1, 1])],
                  [1 / 4 * (1 - pte[2, 0]) * (1 - pte[2, 1]),
                   1 / 4 * (1 + pte[2, 0]) * (1 - pte[2, 1]),
                   1 / 4 * (1 + pte[2, 0]) * (1 + pte[2, 1]),
                   1 / 4 * (1 - pte[2, 0]) * (1 + pte[2, 1])],
                  [1 / 4 * (1 - pte[3, 0]) * (1 - pte[3, 1]),
                   1 / 4 * (1 + pte[3, 0]) * (1 - pte[3, 1]),
                   1 / 4 * (1 + pte[3, 0]) * (1 + pte[3, 1]),
                   1 / 4 * (1 - pte[3, 0]) * (1 + pte[3, 1])]])
    return Q


if __name__ == '__main__':
    Q = matrix_gp2node(1 / np.sqrt(3))
    print(Q, 1 + np.sqrt(3) / 2, 1 - np.sqrt(3) / 2)
    # [[ 1.8660254 -0.5        0.1339746 -0.5      ]
    #  [-0.5        1.8660254 -0.5        0.1339746]
    #  [ 0.1339746 -0.5        1.8660254 -0.5      ]
    #  [-0.5        0.1339746 -0.5        1.8660254]] 1.86602540378 0.133974216
