"""filter results for plotting"""
import numpy as np


def results_at_fixed_y(u, y=0, decimals=3):
    """filter results for a fixed y coordinate

    Args:
        u (numpy array): shape (num_nodes, 4) with format
            [x, y, field_x, field_y].
        y (float, default=0.): y coordinate to filter results
        decimals (int, default=1): number of decimals to approximate
            results at. Example: decilamel=1 implies finding nodes
            coordinates which y=0 +- 0.1.

    Returns:
        filtered_u (numpy array) same shape as u

    """
    # nodes where y = 0
    nodes_index = np.where(np.round(u[:, 1], decimals) == y)[0]
    filtered_u = u[nodes_index]
    # sort filtered_u based on x coordinate
    filtered_u = filtered_u[np.argsort(filtered_u[:, 0])]
    return filtered_u
