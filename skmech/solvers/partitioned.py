"""Solve partitioned system for the incremental problem"""
import numpy as np


def solve_partitioned(model, K_T, f_int, f_ext, increment, k):
    """Solve partitioned system

    Obtain Newton correction for free degree's of freedom and obtain residual
    for restrained degree's of freedom

    [ Kff Kfr ] [ delta_u_f ] = - [ residual_f ]
    [ Krf Krr ] [ delta_u_r ] = - [ residual_r ]

    Returns
    -------
    delta_u, f_int_r
        newton correction and internal force load for restrained dofs

    Note
    ----
    Considering homogeneous Dirichlet boundary conditions, zero displacement
    on the restrained dofs.
    If nonhomogeneus Dirichlet boundary conditions are imposed, then, only in
    the first iteration of each time step they will be enforced

    See (Borst 2012 Section 2.5)

    """
    # Compute residual
    if model.imposed_displ is not None:
        f, r = model.update_free_restrained_dof(increment)
    else:
        # incidences of free and restrained dofs
        f, r = model.id_f, model.id_r
    ff = np.ix_(f, f)
    rf = np.ix_(r, f)
    fr = np.ix_(f, r)
    rr = np.ix_(r, r)

    residual = f_int - f_ext

    # updtade vector with all dofs, the restrained dofs correction is zero
    delta_u = np.zeros(model.num_dof)

    # impose displacement for first iteration of each increment
    if k == 0:
        delta_u[r] = set_imposed_displacement(model, increment, r)
        # solve for free considering non zero restrained correction
        delta_u[f] = - np.linalg.solve(K_T[ff],
                                       residual[f] + K_T[fr] @ delta_u[r])
        residual[r] = - K_T[rf] @ delta_u[f] - K_T[rr] @ delta_u[r]
    else:
        # now all restrained dofs have zero displacement correction
        # solve for free dofs when correction for restrained is zero
        delta_u[f] = np.linalg.solve(K_T[ff], - residual[f])
        # update residual vector with the restrained part
        residual[r] = - K_T[rf] @ delta_u[f]

    # add reaction to external load vector
    f_ext[r] = f_int[r] - residual[r]

    return delta_u, f_ext


def set_imposed_displacement(model, increment, r):
    """Set the imposed displacement for this pseudo-time increment

    Parameters
    ----------
    model.impoed_displ[increment] : dict
        Imposed displacement {physical_tag: (imposed_u_x, imposed_u_y)}
    """
    delta_u = np.zeros(model.num_dof)

    if model.imposed_displ is not None:
        # displacement_location and displacement_value
        for d_loc, d_value in model.imposed_displ[increment].items():
            physical_element = model.get_physical_element(d_loc)
            for eid, [etype, *edata] in physical_element.items():
                # physical points
                if etype == 15:
                    node = edata[-1]  # last entry
                    dof = np.array(model.nodes_dof[node]) - 1
                    if d_value[0] is not None:
                        delta_u[dof[0]] = d_value[0]
                    if d_value[1] is not None:
                        delta_u[dof[1]] = d_value[1]
                # physical lines
                if etype == 1:
                    node_1, node_2 = edata[-2], edata[-1]
                    dof_n1 = np.array(model.nodes_dof[node_1]) - 1
                    dof_n2 = np.array(model.nodes_dof[node_2]) - 1
                    if d_value[0] is not None:
                        # modify dof in x for node 1
                        delta_u[dof_n1[0]] = d_value[0]
                        delta_u[dof_n2[0]] = d_value[0]
                    if d_value[1] is not None:
                        # modify dof in y for node 1
                        delta_u[dof_n1[1]] = d_value[1]
                        # modify dof in y for node 2
                        delta_u[dof_n2[1]] = d_value[1]
    return delta_u[r]
