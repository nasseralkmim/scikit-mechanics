"""plot surface labels"""


def surfaces2dlabel(xyz, surf, physical_surf, line, line_loop, ax):
    """plot surface labels from surf dictionary

    The surf dictionary has the form,

    Example:
    >>> surf = {9: 8}           # surface 9, line_loop 8
    >>> physical_surf = {10: 9}  # physical surface 10, surface 9
    >>> line = {0: [0, 1],      # line 0, nodes 0 and 1
                1: [1, 2],      # line 1, nodes 1 and 2
                2: [2, 3],
                3: [3, 0]}
    >>> line_loop = {8, [0, 1, 2, 3]}  # line_loop 8, lines 0, 1, 2, 3

    where 9 is the physical tag for the surface, and 8 is the tag
    for the line loop that generated the surface. The materials are
    generally applied to the surf tag, in this case 9.

    Args:
        xyz (numpy array): nodes coordinates
        surf (dict): [surf_tag, line_loop_tag]
        line (dict): {line_tag: [n1, n2]}  element nodes in each line
        line_loop (dict): {line_loop_tag: [line_tag1, line_tag2, ...]} lines
            that define a surface
        ax : matplotlib axes object
    """
    for phy_surf, s in physical_surf.items():
        x_m, y_m = 0, 0
        for l in line_loop[surf[s]]:
            nodes = line[l]
            x_m += sum(xyz[nodes, 0])
            y_m += sum(xyz[nodes, 1])

        x_center = x_m/(2*len(line_loop[surf[s]]))
        y_center = y_m/(2*len(line_loop[surf[s]]))
        ax.annotate(s, (x_center, y_center))
