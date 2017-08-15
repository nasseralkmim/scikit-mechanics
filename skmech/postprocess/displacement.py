"""post process displacement
"""


def update_nodes_coordinates(model, u):
    """Update nodes cooridnates

    Args:
        model (obj): object with model attributes
        u (numpy array shape (num_dof, )): displacement field at the degree's
            of freedom.

    Returns:
        numpy array shape (num_nodes, 2): new nodes coordinates
    """
    pass
