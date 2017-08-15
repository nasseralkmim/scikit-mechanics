"""Post process displacement resulting considering the enrichment
"""
import numpy as np
from ..constructor import constructor


def internode_enrichment(U, model, grid=5):
    """Post process enriched results in inter nodes domain

    Args:
        U (numpy array): shape (num_dof,) the displacement at the degree's of
            freedom (dof) appended by the value of the constants referent to
            enriched dofs.
        model (obj): model object

    Returns:
        numpy array: shape (num_nodes, 4) coordinates (x, y) and displacement
             x, y for nodes in mesh.

    Note:
        Considering an enriched element, then

            u(X, t) = u_std + u_enr

        where, u_std is the standard FEM solution and u_enr the enrichment
        part. The enrichment part is given by,

            u_enr(X, t) = \sum_{j=1}^{num_enr_nodes} N_j(Xi) PSI_j(Xi) a_j

        where N_j(Xi) is the standard shape function PSI_j(Xi) is the shifted
        enrichment function, which used the same shape functions therefore it
        is also a function of the isoparametric coordinates. For instance,
        a weak discontinuity enrichment function for node `j` in a element:

            PSI_j = abs(N(Xi) @ phi) - abs(phi[j])

        where, phi = [phi_1, phi_2, phi_3, phi_4] is the signed distance
        function using local index, phi_1 is the first node in element.conn.
        N(Xi) in this case is a 1d array with shape functions for each element
        node and `N @ phi` is an approximation for phi using the standard shape
        functions

    Note:
        The inter node approximation is done by subdividing the element
        in the isoparametric domain in a grid n x n, then compute the
        approximation at all the grid points.
        The approximation is stored in an array together with the point
        coordinates that is transformed from the isoparametric to Cartesian
        system.

    Note:
        model.enriched_elements = [enr_ele1, enr_ele2, ...]: global element tag
            Example: [0, 1, 3]
        model.CONN = [[node1, node2, node3 ...], ...]: global tag of nodes that
            form an element.
        element.enriched_nodes = [node1, node2, ...]: global node tag of nodes
            in this element that are enriched.
        element.dof = [dof1, dof2, ...]: degree's of freedom, standard and
            enriched, for this element.

    """
    # Seprate constant for standard and enriched dof
    Ustd = U[:model.num_std_dof]

    # put nodes results in x, y, field_x, field_y
    nu = dof2nodes(Ustd, model)
    # initialize inter node results (inu)
    inu = []
    uenr = []
    # Loop over each element to access shape functions
    for e, conn in enumerate(model.CONN):
        element = constructor(e, model)

        # fem solution constants a = [Ustd Uenr]
        a = U[element.dof]
        aenr = a[8:]

        # check if element has enriched nodes
        if e in model.enriched_elements:
            # extra grid points inside element isoparamtric coordinates
            xi_grid = np.linspace(-1, 1, grid)
            eta_grid = np.linspace(-1, 1, grid)

            # loop over inter node grid points
            for xi in xi_grid:
                for eta in eta_grid:
                    # compute std shape function at this points
                    # N = [N1, N2, N3, N4]
                    N, _ = element.shape_function([xi, eta])

                    # Nstd shape (2x8) shape function matrix
                    N_s = []
                    for j in range(element.num_std_nodes):
                        N_s.append(np.array([[N[j], 0],
                                             [0, N[j]]]))
                    Nstd = np.block([N_s[i]
                                     for i
                                     in range(element.num_std_nodes)])

                    # Shape function matrix for this zls
                    Nenr_zls = {}
                    # loop for each zero level set
                    for ind, zls in enumerate(element.zerolevelset):
                        # signed distance for nodes in this element for  zls
                        phi = zls.phi[element.conn]  # phi with local index

                        # assemble the enriched shaped funcion matrix
                        Nk = {}
                        # loop in enriched nodes (global tag of nodes)
                        for n in element.enriched_nodes[ind]:
                            j = element.global2local_index(n)
                            # enrichment function for weak discontinuity
                            # TODO: abstract that as an element attribute
                            psi = abs(N @ phi) - abs(phi[j])

                            Nk[n] = np.array([[N[j]*psi, 0],
                                              [0, N[j]*psi]])

                        # Nenr matrix
                        # Nenr = [Nenr1, Nenr2, ...] number of enriched nodes
                        Nenr_zls[ind] = np.block([Nk[i]
                                                  for i in
                                                  element.enriched_nodes[ind]])

                    Nenr = np.block([Nenr_zls[i]
                                     for i in
                                     range(len(element.zerolevelset))])
                    # assemble enhanced shape function
                    Nenh = np.block([Nstd, Nenr])

                    # map back to cartesian coordinates
                    x, y = element.mapping(N, element.xyz)

                    # compute approximation in [xi, eta]
                    u_ = Nenh @ a
                    inu.append([x, y, u_[0], u_[1]])

                    ue_ = Nenr @ aenr
                    uenr.append([x, y, ue_[0], ue_[1]])

    u = np.block([[nu],
                  [np.array(inu)]])

    return u, np.array(uenr)


def dof2nodes(field, model):
    """convert nodal result to coordinate base

    The array field is aligned with dof index, this function converts it to
    coordinate base. For example,

       dof0 --> field[0]
       dof1 --> field[1]

    is converted to:

        [[(x_1, y_1), (field_x_2, field_y_2)],
         [(x_2, y_2), (field_x_2, field_y_2)]]

    So we can add intermediate inter node values by specifying its coordinate.

    Args:
        field (numpy array): nodal field values
        model (obj): model object

    Returns:
        (numpy array) shape (n, 4) with [(x, y), (field_x, field_y)] the field
            value for each coordinate

    Example:
        field = [1, 2, 3, 4]
        model.XYZ = [[0, 0], [0, 1]]

        >>> dof2nodes(field, model)
        [[(0, 0), (1, 2]), [(0, 1), (3, 4)]]
    """
    u = []
    for n, xyz in enumerate(model.XYZ):
        dof = model.nodal_DOF[n]
        u.append([xyz[0], xyz[1], field[dof[0]], field[dof[1]]])
    return np.array(u)
