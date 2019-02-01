"""plot boundary lines label"""


def boundarylines2dlabel(xyz, nodes_in_bound_line, ax):
    """plot labels for boundary lines

    The numpy array nodes_in_bound_line has the form:

    Example:
    >>> nodes_in_bound_line = [[0 0 1]
                               [0 1 4]
                               [0 4 5]
                               [1 5 6]
                               [2 6 7]
                               [2 7 2]
                               [2 2 3]
                               [3 3 0]]

    Where the first number indicates the line tag, and the 2nd
    and 3rd the nodes of the element in that boundary line.

    Args:
        xyz (numpy array): nodes coordinates
        nodes_in_bound_line (numpy array): [line_tag, node1_tag, node2_tag]
        ax : matplotlib axes object

    """
    for line, n1, n2 in nodes_in_bound_line:
        x, y = sum(xyz[[n1, n2], 0])/2, sum(xyz[[n1, n2], 1])/2
        ax.annotate(line, (x, y))
