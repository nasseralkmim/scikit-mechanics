"""writes the gmsh file"""


def write(file_name, node_data, data_label):
    """Write gmsh node data

    Parameters
    ----------
    file_name : str
    node_data : dict
        dictionary with node tag and field as an array like
    data_label : str

    """
    with open(file_name, 'a') as gmsh_file:
        gmsh_file.write('$NodeData\n'
                        '1\n"{}"\n1\n0.0\n3\n0\n{}\n{}\n'
                        .format(data_label, 3,
                                len(node_data)))
        for nid, ndata in node_data.items():
            gmsh_file.write(str(nid))
            for data in ndata:
                gmsh_file.write(' {}'.format(str(data)))
            gmsh_file.write(' 0\n')
        gmsh_file.write('$EndNodeData')


if __name__ == '__main__':
    node_data = {1: [0, 0], 2: [1, 0], 3: [.5, 0]}
    write('patch.msh', node_data, 'Displacement')

