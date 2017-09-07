"""plot elements with nodes coordinates and connectivity"""
from .line2d import line2d
import numpy as np


def elements2d(nodes, elements, ax, color='k'):
    """plot elements lines

    Parameters
    ----------
    nodes : dict {nid, [x, y, z]}
    elements : dict {eid, [edata]}
    ax : matplotlib axes

    """
    for eid, [etype, _, _, _, *ele_nodes] in elements.items():
        enodes = np.append(ele_nodes, ele_nodes[0])  # complete cicle
        for n1, n2 in zip(enodes[:-1], enodes[1:]):
            line2d([nodes[n1][0], nodes[n2][0]],
                   [nodes[n1][1], nodes[n2][1]], ax,
                   color=color)
