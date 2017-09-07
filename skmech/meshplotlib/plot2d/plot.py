"""plot mesh from nodal coordinates and connectivity"""
import numpy as np
import matplotlib.pyplot as plt
from .nodes2d import nodes2d
from .elements2d import elements2d
from .nodes2dlabel import nodes2dlabel
from .elements2dlabel import elements2dlabel
from .boundarylines2dlabel import boundarylines2dlabel
from .surfaces2dlabel import surfaces2dlabel
from .field2d import field2d


def geometry2(model, ax,
              elements=True,
              nodes=False,
              nodes_label=False,
              elements_label=False,
              surfaces_label=False,
              boundary_label=False):
    """plot geometry

    Args:
        model (obj): object with model geometry parameters
        ax (obj): matplotlib axes object
    """

    try:
        if nodes is True:
            xyz = model.XYZ
            nodes2d(xyz, ax)
        if nodes_label is True or isinstance(nodes_label, list):
            xyz = model.XYZ
            nodes2dlabel(xyz, ax, nodes_label)

        if elements is True:
            xyz = model.XYZ
            conn = model.CONN
            elements2d(xyz, conn, ax)
        if elements_label is True or isinstance(elements_label, list):
            xyz = model.XYZ
            conn = model.CONN
            elements2dlabel(xyz, conn, ax, elements_label)

        if surfaces_label is True:
            xyz = model.XYZ
            surf = model.surf
            physical_surf = model.physical_surf
            line = model.line
            line_loop = model.line_loop
            surfaces2dlabel(xyz, surf, physical_surf, line, line_loop, ax)

        if boundary_label is True:
            xyz = model.XYZ
            nodes_in_bound_line = model.nodes_in_bound_line
            boundarylines2dlabel(xyz, nodes_in_bound_line, ax)
    except AttributeError:
        print('Model object should be created with gmsh.Parse attributes!')

    # recompute the ax.dataLim
    ax.relim()
    # update ax.viewLim using the new dataLim
    ax.autoscale_view()
    ax.set_aspect('equal')


def field(xyz, field, ax, orientation='vertical',
          cbar_label='Stress', **kwargs):
    """plot field

    """
    field2d(xyz, field, ax, orientation, cbar_label, **kwargs)
    ax.set_aspect('equal')


def deformed(nodes, elements, displ, ax, magf=1):
    """plot deformed structure

    Parameters
    ----------
    nodes : dict {nid, [x, y, z]}
    elements : dict {eid, [edata]}
    displ : dict {nid, [dx, dy]}

    """
    nodes_updt = {}
    for nid, xyz in nodes.items():
        nodes_updt[nid] = [xyz[0] + displ[nid][0]*magf,
                           xyz[1] + displ[nid][1]*magf]

    elements2d(nodes_updt, elements, ax, color='red')
    # recompute the ax.dataLim
    ax.relim()
    # update ax.viewLim using the new dataLim
    ax.autoscale_view()
    ax.set_aspect('equal')


def geometry(nodes, elements, ax):
    """plot geometry
    """
    elements2d(nodes, elements, ax, color='k')
    # recompute the ax.dataLim
    ax.relim()
    # update ax.viewLim using the new dataLim
    ax.autoscale_view()
    ax.set_aspect('equal')


if __name__ == '__main__':
    class Model:
        pass
    model = Model
    model.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                          [2, 0], [2, 1]])
    model.CONN = np.array([[0, 1, 2, 3],
                           [1, 4, 5, 2]])
    fig, ax = plt.subplots()
    geometry(model, ax, elements=True, nodes_label=True)

    displ = np.array([[1, 0, .2, 0],
                      [1, 1, .3, 0],
                      [0, 1, .1, 0],
                      [0, 0, .1, 0],
                      [2, 0, .4, 0],
                      [2, .9999994, .5, 0]])
    deformed(model, displ, ax)
    plt.show()
