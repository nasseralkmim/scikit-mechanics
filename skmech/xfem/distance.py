"""computes distance from mesh nodes to zero level set"""
import numpy as np
from scipy import interpolate
import skfmm


def distance(zero_ls, grid_x, grid_y, xyz):
    """Computes distance from zero level set and nodes coordinates

    Interpolates the data from the distance grid and evaluate
     that function at mesh nodes.

    Parameters
    ----------
    zero_ls : numpy array
        2d array whose contour level 0 defines
        the desirable interface
    grid_x, grid_y 2d numpy arrays
        grid coordinates
    xyz numpy array shape(nn, D)
        2nd array with nodes coordinates
        (x, y), nn is the number of nodes and D is dimension (D=2)

    Returns
    -------
    numpy array shape(nn,)
        with distance from mesh nodes to zero level set.

    """
    # number of division in each dimension
    dx, dy = np.size(zero_ls[:, 0]), np.size(zero_ls[0, :])
    # dx is the cell length in each direction
    try:
        dist = skfmm.distance(zero_ls, dx=[1 / (dx - 1), 1 / (dy - 1)])
    except ValueError:
        raise Exception('Adjust the grid resolution of the zero level set'
                        ' or function does not have a zero level set')

    # values shape (n,) at points shape (n, D) D is dimensions
    values = np.ndarray.flatten(dist)
    # points showld have shape (n, D) n is the number of samples
    points = np.vstack((np.ndarray.flatten(grid_x),
                        np.ndarray.flatten(grid_y))).T
    phi = interpolate.griddata(points, values, xyz)

    # substituve nan values to 0
    phi[np.isnan(phi)] = 0.

    return phi
