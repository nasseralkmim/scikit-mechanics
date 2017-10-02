"""Plot 2d field
"""
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
    cax = ax.tricontourf(trian, field, rasterized=True, **kwargs)
    cbar = plt.colorbar(cax, orientation=orientation)
    cbar.set_label(cbar_label)
