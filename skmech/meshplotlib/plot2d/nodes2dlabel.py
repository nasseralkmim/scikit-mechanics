"""plot nodes label in 2d plot"""


def nodes2dlabel(xyz, ax, nodes_label, **kwargs):
    """plot 2d nodes labels from xyz array in ax

    Args:
        xyz (narray): nodes coordinates
        ax : matplotlib axes object
        nodes_label (boolean or list): if boolean plot all nodes
            labels, if a list plot olnly labels for nodes in the list
    """
    if nodes_label is True:
        for i, [x, y] in enumerate(xyz):
            ax.annotate(i, (x, y))
    else:
        for n in nodes_label:
            ax.annotate(n, (xyz[n, 0], xyz[n, 1]))
