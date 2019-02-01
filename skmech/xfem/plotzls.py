"""plot 2d level set"""
import matplotlib.pyplot as plt
import numpy as np


def plot_zls(zls, ax, plot_cbar=True, factor=1, **kwargs):
    """plot 2d level set with zlsmask

    Parameters
    ----------
    zls : obj or list of obj
        object that contains grid X, Y and mask_zls
    ax : matplotlib axes instance

    """
    if isinstance(zls, list):
        mask = np.zeros_like(zls[0].mask)
        for z in zls:
            mask += z.mask
        mask = (mask / len(zls) - .5) / .5  # 0,1 -> -1, 1
        zlsmask = mask
        X, Y = zls[0].grid_x, zls[0].grid_y
    else:
        zlsmask = zls.mask
        X, Y = zls.grid_x, zls.grid_y

    # ax.contour(X * factor, Y * factor,
    #            zlsmask, levels=[0], **kwargs)
    c = ax.contourf(X * factor, Y * factor, zlsmask)
    if plot_cbar:
        cbar = plt.colorbar(c)
        if np.max(zlsmask) == 1 and np.min(zlsmask) == -1:
            cbar.set_label('Zero level set mask')
        else:
            cbar.set_label(r'Zero level set $\phi(x)$, (m)')
