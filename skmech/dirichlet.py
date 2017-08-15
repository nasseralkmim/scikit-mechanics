"""Apply the boundary conditions to matrices and vectors"""
import numpy as np


def dirichlet(K, F, model):
    """Apply Dirichlet BC.

    Parameters
    ----------
    K : numpy array shape(num_dof, num_dof)
    F : numpy array shape(num_dof,)
    displacement : dict or None
        displacement boundary condition in a dictionary with key the physical
        element where the boundary is.

    Returns
    -------
    K : numpy array
        Modified array to ensure boundary condition
    F : numpy array
        Modifiec array

    """
    if model.displacement is not None:
        for d_location, d_vector in model.displacement.items():
            physical_element = model.get_physical_element(d_location)
            if len(physical_element) == 0:
                raise Exception('Check if the physical element {} '
                                'was defined in gmsh'.format(physical_element))
            for eid, [etype, *edata] in physical_element.items():
                # physical points
                if etype == 15:
                    node = edata[-1]  # last entry
                    dof = np.array(model.nodes_dof[node]) - 1
                    K[dof[1], dof[1]] = 1

                    if d_vector[0] is not None:
                        K[dof[0], :] = 0  # zero lines
                        K[dof[0], dof[0]] = 1  # diagonal equal 1
                        F[dof[0]] = d_vector[0]
                    if d_vector[1] is not None:
                        K[dof[1], :] = 0  # zero lines
                        K[dof[1], dof[1]] = 1  # diagonal equal 1
                        F[dof[1]] = d_vector[1]
    return K, F
