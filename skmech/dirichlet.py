"""Apply the boundary conditions to matrices and vectors"""
import numpy as np
from .constructor import constructor


def dirichlet(K, F, model):
    """Apply Dirichlet BC.

    Parameters
    ----------
    K : ndarray shape((num_dof, num_dof))
    F : ndarray shape((num_dof,))
    model : Model object

    Returns
    -------
    K : numpy array
        Modified array to ensure boundary condition
    F : numpy array
        Modified array

    """
    # make a copy of arrays to not modify the original ones
    K = np.copy(K)
    F = np.copy(F)
    if model.displacement_bc is not None:
        for d_location, d_vector in model.displacement_bc.items():
            physical_element = model.get_physical_element(d_location)
            if len(physical_element) == 0:
                raise Exception('Check if the physical element {} '
                                'was defined in gmsh'.format(physical_element))
            for eid, [etype, *edata] in physical_element.items():
                # physical points
                if etype == 15:
                    node = edata[-1]  # last entry
                    dof = np.array(model.nodes_dof[node]) - 1
                    if d_vector[0] is not None:
                        F -= K[:, dof[0]] * d_vector[0]
                        F[dof[0]] = d_vector[0]
                        K[dof[0], :] = 0  # zero lines
                        K[:, dof[0]] = 0  # zero column
                        K[dof[0], dof[0]] = 1  # diagonal equal 1
                    if d_vector[1] is not None:
                        F -= K[:, dof[1]] * d_vector[1]
                        K[dof[1], :] = 0
                        K[:, dof[1]] = 0
                        K[dof[1], dof[1]] = 1  # diagonal equal 1
                        F[dof[1]] = d_vector[1]
                # physical lines
                if etype == 1:
                    node_1, node_2 = edata[-2], edata[-1]
                    dof_n1 = np.array(model.nodes_dof[node_1]) - 1
                    dof_n2 = np.array(model.nodes_dof[node_2]) - 1
                    if d_vector[0] is not None:
                        # modify dof in x for node 1
                        F -= K[:, dof_n1[0]] * d_vector[0]
                        F[dof_n1[0]] = d_vector[0]
                        K[dof_n1[0], :] = 0  # zero lines
                        K[:, dof_n1[0]] = 0  # zero column
                        K[dof_n1[0], dof_n1[0]] = 1  # diagonal equal 1
                        # modify dof in x for node 2
                        F -= K[:, dof_n2[0]] * d_vector[0]
                        F[dof_n2[0]] = d_vector[0]
                        K[dof_n2[0], :] = 0  # zero lines
                        K[:, dof_n2[0]] = 0  # zero column
                        K[dof_n2[0], dof_n2[0]] = 1  # diagonal equal 1
                    if d_vector[1] is not None:
                        # modify dof in y for node 1
                        F -= K[:, dof_n1[1]] * d_vector[1]
                        K[dof_n1[1], :] = 0
                        K[:, dof_n1[1]] = 0
                        K[dof_n1[1], dof_n1[1]] = 1  # diagonal equal 1
                        F[dof_n1[1]] = d_vector[1]
                        # modify dof in y for node 2
                        F -= K[:, dof_n2[1]] * d_vector[1]
                        K[dof_n2[1], :] = 0
                        K[:, dof_n2[1]] = 0
                        K[dof_n2[1], dof_n2[1]] = 1  # diagonal equal 1
                        F[dof_n2[1]] = d_vector[1]
    return K, F


def imposed_displacement(model):
    """Create load vector due imposed displacement"""

    try:
        num_dof = model.num_dof
    except AttributeError:
        raise Exception('Model object does not have num_dof attribute')

    # vector with imposed displacement in the dof slot
    imposed_u = np.zeros(num_dof)
    for d_location, d_vector in model.imposed_displ.items():
        physical_element = model.get_physical_element(d_location)
        for eid, [etype, *edata] in physical_element.items():
            if etype != 1:
                raise Exception('Imposed displacement only on lines (type 1)')
            # last two items in element data
            node_1, node_2 = edata[-2], edata[-1]
            # subtract 1 because python starts at 0
            dof_n1 = np.array(model.nodes_dof[node_1]) - 1
            dof_n2 = np.array(model.nodes_dof[node_2]) - 1
            if d_vector[0] is not None:
                # first dof of n1 and n2 in this physical line
                imposed_u[dof_n1[0]] = d_vector[0]
                imposed_u[dof_n2[0]] = d_vector[0]
            if d_vector[1] is not None:
                # first dof of n1 and n2 in this physical line
                imposed_u[dof_n1[1]] = d_vector[1]
                imposed_u[dof_n2[1]] = d_vector[1]

    Pd = np.zeros(num_dof)

    # Loop over elements
    for eid, [etype, *edata] in model.elements.items():
        # create element object
        element = constructor(eid, etype, model)
        # recover element nodal displacement increment,  shape (8,)
        dof = np.array(element.dof) - 1  # numpy starts at 0

        u_ele = imposed_u[dof]

        Pd_ele = np.zeros(8)
        # loop over gauss points
        for gp_id, [w, gp] in enumerate(zip(element.gauss.weights,
                                            element.gauss.points)):
            # build element strain-displacement matrix shape (3, 8)
            N, dN_ei = element.shape_function(xez=gp)
            dJ, dN_xi, _ = element.jacobian(element.xyz, dN_ei)
            B = element.gradient_operator(dN_xi)

            D = element.c_matrix(N)

            # interate over element
            Pd_ele += B.T @ D @ B @ u_ele * (dJ * w * element.thickness)

        # += because elements can share same dof
        Pd[element.id_v] += Pd_ele

    return Pd


if __name__ == '__main__':
    import skmech
    class Mesh():
        pass
    # 4 element with offset center node
    msh = Mesh()
    msh.nodes = {
        1: [0, 0, 0],
        2: [1, 0, 0],
        3: [1, 1, 0],
        4: [0, 1, 0],
        5: [.5, 0, 0],
        6: [1, .5, 0],
        7: [.5, 1, 0],
        8: [0, .5, 0],
        9: [.4, .6]
    }
    msh.elements = {
        1: [15, 2, 12, 1, 1],
        2: [15, 2, 13, 2, 2],
        3: [1, 2, 7, 2, 2, 6],
        4: [1, 2, 7, 2, 6, 3],
        7: [1, 2, 5, 4, 4, 8],
        8: [1, 2, 5, 4, 8, 1],
        9: [3, 2, 11, 10, 1, 5, 9, 8],
        10: [3, 2, 11, 10, 5, 2, 6, 9],
        11: [3, 2, 11, 10, 9, 6, 3, 7],
        12: [3, 2, 11, 10, 8, 9, 7, 4]
    }
    material = skmech.Material(E={11: 10000}, nu={11: 0.3})
    support = {12: (0, 0), 13: (None, 0)}
    imposed_displ = {7: (.01, None)}
    model = skmech.Model(
        msh,
        material=material,
        imposed_displ=imposed_displ,
        displacement_bc=support,
        num_quad_points=2)
    print(imposed_displacement(model))
