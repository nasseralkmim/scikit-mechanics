"""creates an object with the zero level set definition"""
import numpy as np


class Create(object):
    """creates the zero level set object with its attributes

    Args:
        region (func(x, y) or list of tuples): region that defines the level
            set the zero level set defines the discontinuity interface. Or
            list of tuples with (xc, yc, r) of circles.
        x_domain (list): domian defined by (x_domain[0], x_domain[1])
        y_domain (list): same as x
        num_div (float, optional): number of division for defining the
            level set, the greater the value more precisa the interface
            will be defined

    Attributes:
        grid_x (numpy array): 2d array shape (num_div, num_div)
            with grid value for x direction
        grid_y (numpy array): same as x for y direction
        mask_ls (numpy array): 2d array shape (num_div, num_div) with
            -1 and 1 where the points between these two values define the
            discontinuity interface.

    """
    def __init__(self, region, x_domain, y_domain, num_div=50,
                 matrix=1, reinforcement=-1, material=None):
        self.grid_x, self.grid_y = np.meshgrid(np.linspace(x_domain[0],
                                                           x_domain[1],
                                                           num_div),
                                               np.linspace(y_domain[0],
                                                           y_domain[1],
                                                           num_div))
        # flip the grid so it agrees with cartesian coordinates
        self.grid_x = np.flipud(self.grid_x)
        self.grid_y = np.flipud(self.grid_y)

        ls = 0
        if callable(region):
            ls = region(self.grid_x, self.grid_y)
        elif type(region) is list:
            for xc, yc, r in region:
                mask = (self.grid_x - xc)**2 + (self.grid_y - yc)**2 <= r**2
                ls += mask.astype(int)
            ls = (-ls + 1/2)*2

        # be careful with 0 division, converting it to 0.
        with np.errstate(divide='ignore', invalid='ignore'):
            self.mask_ls = np.nan_to_num(ls/abs(ls))

        if material is not None:
            self.material = material


if __name__ == '__main__':
    # def func(x, y):
    #     return (x - 2)**2 + (y - 2)**2 - 1.8**2
    # z_ls = Create(func, [0, 2], [0, 2], num_div=3)
    # print(z_ls.mask_ls)
    # [[ 1.  1.  1.]
    #  [ 1. -1. -1.]
    #  [ 1. -1. -1.]]
    # z_ls = Create([(1, 1, .2),
    #                (2, 2, .2),
    #                (.2, 1.5, .5)],
    #               [0, 2], [0, 2], num_div=10)
    # print(z_ls.mask_ls)
    # [[ 1.  1.  1.  1.  1.  1.  1.  1.  1. -1.]
    #  [-1. -1. -1.  1.  1.  1.  1.  1.  1.  1.]
    #  [-1. -1. -1. -1.  1.  1.  1.  1.  1.  1.]
    #  [-1. -1. -1. -1.  1.  1.  1.  1.  1.  1.]
    #  [-1. -1. -1.  1. -1. -1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1. -1. -1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]
    #  [ 1.  1.  1.  1.  1.  1.  1.  1.  1.  1.]]
    samplesize = 10
    np.random.seed(1)
    r = np.random.randn(samplesize) / 10.  # normal dist
    c = np.random.uniform(0, 2, size=(samplesize, 2))
    region = [(xc, yc, ri) for [xc, yc], ri in zip(c, r)]
    z_ls = Create(region, [0, 2], [0, 2], num_div=10)
    print(z_ls.mask_ls)
    import matplotlib.pyplot as plt
    plt.imshow(z_ls.mask_ls)
    plt.show()
