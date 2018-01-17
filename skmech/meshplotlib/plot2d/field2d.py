"""Plot 2d field
"""
import numpy as np
import matplotlib.tri as tri
import matplotlib.pyplot as plt


def field2d(xyz, field, ax, orientation, cbar_label, **kwargs):
    """Plot 2d scalar field

    Parameters
    ----------
    xyz : ndarray, shape(num_points, 2))
        nodes coordinates (x, y)
    field : ndarray, shape(num_points, ))
        scalar field value at nodes
    ax : matplotlib axes object

    """
    trian = tri.Triangulation(xyz[:, 0], xyz[:, 1])
    cax = ax.tricontourf(trian, field, **kwargs)
    cbar = plt.colorbar(cax, orientation=orientation)
    cbar.set_label(cbar_label)
    return None


def field2d_nodes(xyz, conn, field, ax, orientation, cbar_label, **kwargs):
    """Plot 2d scalar field giving triangulation

    Parameters
    ----------
    xyz : ndarray, shape(num_points, 2))
        nodes coordinates (x, y)
    conn : ndarray, shape(num_elements, 4)
        connectivity array for quad element
    field : ndarray, shape(num_points, ))
        scalar field value at nodes
    ax : matplotlib axes object

    Note
    ----
    Only for quad element

    """
    # TODO: make this more efficient
    trian = []
    for n1, n2, n3, n4 in conn:
        trian.append([n1, n2, n3])
        trian.append([n1, n3, n4])
    trian = np.array(trian)
  
    cax = ax.tricontourf(xyz[:, 0], xyz[:, 1], trian, field, **kwargs)
    cbar = plt.colorbar(cax, orientation=orientation)
    cbar.set_label(cbar_label)
    return None


if __name__ == '__main__':
    xyz = np.array([[0, 0],     # 1
                    [1, 0],     # 2
                    [1, .3],    # 3
                    [1, 1],     # 4
                    [.7, 1.1],  # 5
                    [.7, .5]])  # 6
    field = np.array([3, 3, 3, 3, 3, 4])
    conn = np.array([[1, 2, 3, 6],
                     [6, 3, 4, 5]]) - 1
    fig, ax = plt.subplots()
    field2d(xyz, field, ax, orientation='vertical', cbar_label='None')
    ax.plot(xyz[:, 0], xyz[:, 1], 'rx')

    fig, ax = plt.subplots()
    field2d_nodes(xyz, conn, field, ax, orientation='vertical',
                  cbar_label='None')
    ax.plot(xyz[:, 0], xyz[:, 1], 'rx')
    plt.show()
