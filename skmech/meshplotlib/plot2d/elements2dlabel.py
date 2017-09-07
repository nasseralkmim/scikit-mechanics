"""plot element label"""


def elements2dlabel(xyz, conn, ax, elements_label, **kwargs):
    """plot 2d element labels

    Args:
        xyz (narray): nodes coordinates
        ax : matplotlib axes object
        elements_label (boolean or list): if boolean plot all element
            labels, if a list plot olnly labels for elements in the list

    """
    if elements_label is True:
        for e, conn in enumerate(conn):
            x_ele_center = sum(xyz[conn, 0])/len(conn)
            y_ele_center = sum(xyz[conn, 1])/len(conn)
            ax.annotate(e, (x_ele_center, y_ele_center))
    else:
        for e in elements_label:
            x_ele_center = sum(xyz[conn[e], 0])/len(conn[e])
            y_ele_center = sum(xyz[conn[e], 1])/len(conn[e])
            ax.annotate(e, (x_ele_center, y_ele_center))
