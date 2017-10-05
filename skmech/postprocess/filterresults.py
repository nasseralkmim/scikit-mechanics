"""filter results for plotting"""
import numpy as np


def displacements_at_fixed_y(u, nodes, y=0, decimals=3):
    """Filter results for a fixed y coordinate

    Parameters
    ----------
    u : dict
        displacement for each node
    nodes : dict
        nodes coordinates
    y : float, default=0.0
        y coordinate to filter results
    decimals : int, default=1
        number of decimals to approximate results at.
        Example: decilamel=1 implies finding nodescoordinates which y=0 +- 0.1.

    Returns
    -------
    numpy array
        array with [x, y, ux, uy]

    """
    u_array = dict2array(u, nodes)
    # nodes where y = 0
    nodes_index = np.where(np.round(u_array[:, 1], decimals) == y)[0]
    filtered_u = u_array[nodes_index]
    # sort filtered_u based on x coordinate
    filtered_u = filtered_u[np.argsort(filtered_u[:, 0])]
    return filtered_u


def field_at_fixed_y(field, nodes, y=0, decimals=3):
    """Filter field at specific y

    Parameters
    ----------
    field : ndarray, shape((num_points, 3))
        field coordinate points and fielf value (x, y, value)
    nodes : dict
        nodes coordinate {nid: [x, y]}

    """
    nodes_index = np.where(np.round(field[:, 1], decimals) == y)[0]
    filtered_field = field[nodes_index]
    # sort filtered_u based on x coordinate
    filtered_field = filtered_field[np.argsort(filtered_field[:, 0])]
    return filtered_field


def field_at_fixed_x(field, nodes, x=0, decimals=3):
    """Filter field at specific y

    Parameters
    ----------
    field : ndarray, shape((num_points, 3))
        field coordinate points and fielf value (x, y, value)
    nodes : dict
        nodes coordinate {nid: [x, y]}

    """
    nodes_index = np.where(np.round(field[:, 0], decimals) == x)[0]
    filtered_field = field[nodes_index]
    # sort filtered_u based on y coordinate
    filtered_field = filtered_field[np.argsort(filtered_field[:, 1])]
    return filtered_field


def dict2array(u, nodes):
    """Convert dictionary to array

    Parameters
    ----------
    u : dict
        dictionary with node index and its displacement {nid, [ux, uy]}
    nodes : dict
        dictionary with node index and its location coordinates

    Returns
    -------
    numpy array
        numpy array with [x, y, ux, uy]

    """
    u_array = np.empty((len(u), 4))
    for i, (nid, [ux, uy]) in enumerate(u.items()):
        u_array[i] = [nodes[nid][0], nodes[nid][1], ux, uy]
    return u_array


if __name__ == '__main__':
    u = {0: [10, 20], 2: [1.1, 2.2]}
    nodes = {0: [0, 0], 2: [0, 1]}
    uar = dict2array(u, nodes)
    print(uar)
