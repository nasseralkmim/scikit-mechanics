"""read gmsh files: .geo and .msh and store the respective variables
"""
import numpy as np


def parse_msh(mesh_file):
    """parse .msh file

    The format is defined in gmsh manual. For nodes and elements is:

        $Nodes
        number-of-nodes
        node-number x-coord y-coord z-coord
        …
        $EndNodes
        $Elements
        number-of-elements
        elm-number elm-type number-of-tags < tag > … node-number-list
        …
        $EndElements

    """
    nodes = {}
    elements = {}

    with open(mesh_file, 'r') as f:
        while True:
            line = f.readline()
            if not line:        # empty line, end of file
                break
            section = line.strip()
            if section == '$Nodes':
                line_nodes = f.readline()
                num_nodes = int(line_nodes)
                for _ in range(num_nodes):
                    data = f.readline()
                    node_index = int(data[0])
                    nodes[node_index] = np.fromstring(data[1:],
                                                      dtype=np.float64,
                                                      sep=' ')
            if section == '$Elements':
                line_ele = f.readline()
                num_ele = int(line_ele)
                for _ in range(num_ele):
                    data = f.readline()
                    tags = np.fromstring(data, dtype=int, sep=' ')
                    ele_index = int(tags[0])
                    elements[ele_index] = tags[1:]
    return nodes, elements


if __name__ is '__main__':
    n, e = parse_msh('../../examples/patch.msh')
    print(e)
