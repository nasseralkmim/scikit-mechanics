"""Plot 2d field
"""
import matplotlib.tri as tri
import matplotlib.pyplot as plt


def field2d(xyz, field, ax, orientation, cbar_label, **kwargs):
    """Plot 2d scalar field

    Args:
        xyz (numpy array shape (num_nodes, 2)): nodes coordinates
        field (numpy array shape (num_nodes, )): scalar field value at nodes
        ax (obj): matplotlib ax

    """
    trian = tri.Triangulation(xyz[:, 0], xyz[:, 1])
    cax = ax.tricontourf(trian, field, **kwargs)
    cbar = plt.colorbar(cax, orientation=orientation)
    cbar.set_label(cbar_label)
