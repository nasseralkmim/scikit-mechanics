"""convert dof results to nodal result with node coordinates
"""
import numpy as np


def dof2node_coord(field, model):
    """convert nodal result to coordinate base

    Parameters
    ----------
    field : numpy array
        nodal field values
    model : Build instance

    Returns
    -------
    numpy array shape (n, 4)
        with [(x, y), (field_x, field_y)] the field
        value for each coordinate

    Note
    ----
    The array field is aligned with dof index, this function converts it to
    coordinate base. For example,

       dof0 --> field[0]
       dof1 --> field[1]

    is converted to:

        [[(x_1, y_1), (field_x_2, field_y_2)],
         [(x_2, y_2), (field_x_2, field_y_2)]]

    So we can add intermediate inter node values by specifying its coordinate.

    Example
    -------
    >>> class Model: pass
    >>> class Mesh: pass
    >>> msh = Mesh()
    >>> msh.nodes = {0: [0, 0], 1:[0, 1]}
    >>> model = Model()
    >>> model.mesh = msh
    >>> field = [10, 20, 30, 40]
    >>> dof2nodes(field, model)
    np.array([[0, 0, 10, 20], [0, 1, 30, 40]])
    """
    u = []
    for nid, xyz in model.mesh.nodes.items():
        dof = np.array(model.nodes_dof[nid]) - 1
        u.append([xyz[0], xyz[1], field[dof[0]], field[dof[1]]])
    return np.array(u)


def dof2node(field, model):
    """dof2node

    Parameters
    ----------
    field : numpy array
    model : object from Model
        must gave nodes from gmsh and nodal dofs attributes

    Returns
    -------
    u : dict
        dictionary with node index and respective displacement

    Example
    -------
    >>> class Model: pass
    >>> class Mesh: pass
    >>> msh = Mesh()
    >>> msh.nodes = {1: [0, 0], 2:[0, 1]}
    >>> model = Model()
    >>> model.mesh = msh
    >>> model.nodes_dof = {1: [1, 2], 2:[3, 4]}
    >>> field = np.array([10, 20, 30, 40])
    >>> u = dof2node(field, model)
    {1: array([10, 20]), 2: array([30, 40])}
    """
    u = {}
    for nid, xyz in model.mesh.nodes.items():
        dof = np.array(model.nodes_dof[nid]) - 1
        u[nid] = field[dof]
    return u


if __name__ == '__main__':
    class Model: pass
    class Mesh: pass
    msh = Mesh()
    msh.nodes = {1: [0, 0], 2:[0, 1]}
    model = Model()
    model.mesh = msh
    model.nodes_dof = {1: [1, 2], 2:[3, 4]}
    field = np.array([10, 20, 30, 40])
    u = dof2node(field, model)
