"""Apply the boundary conditions to matrices and vectors"""
import numpy as np


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
                if etype == 1:
                    node_1, node_2 = edata[-2], edata[-1]
                    dof_1 = np.array(model.nodes_dof[node_1]) - 1
                    dof_2 = np.array(model.nodes_dof[node_2]) - 1
                    if d_vector[0] is not None:
                        # modify dof in x for node 1
                        F -= K[:, dof_1[0]] * d_vector[0]
                        F[dof_1[0]] = d_vector[0]
                        K[dof_1[0], :] = 0  # zero lines
                        K[:, dof_1[0]] = 0  # zero column
                        K[dof_1[0], dof_1[0]] = 1  # diagonal equal 1
                        # modify dof in x for node 2
                        F -= K[:, dof_2[0]] * d_vector[0]
                        F[dof_2[0]] = d_vector[0]
                        K[dof_2[0], :] = 0  # zero lines
                        K[:, dof_2[0]] = 0  # zero column
                        K[dof_2[0], dof_2[0]] = 1  # diagonal equal 1
                    if d_vector[1] is not None:
                        # modify dof in y for node 1
                        F -= K[:, dof_1[1]] * d_vector[1]
                        K[dof_1[1], :] = 0
                        K[:, dof_1[1]] = 0
                        K[dof_1[1], dof_1[1]] = 1  # diagonal equal 1
                        F[dof_1[1]] = d_vector[1]
                        # modify dof in y for node 2
                        F -= K[:, dof_2[1]] * d_vector[1]
                        K[dof_2[1], :] = 0
                        K[:, dof_2[1]] = 0
                        K[dof_2[1], dof_2[1]] = 1  # diagonal equal 1
                        F[dof_2[1]] = d_vector[1]
    return K, F
