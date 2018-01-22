"""Build mesh object
"""
from .gmsh_reader import parse_msh


class Mesh(object):
    """Creates mesh object

    Parameters
    ----------
    nodes : dict
        Nodes coordinate with the format {node_id : array(x, y, z)}
    element : dict
        Elements with format:
            {element_id: array(type, #tags, tag1, ..., n1, n2 ...)}
        type is the gmsh type, #tags is the number of tags, generally is
        referred to physical geometry element tag and the geometry element tag
        n1, n2 ... are the nodes tag that form the element.

    Attributes
    ----------
    nodes : dict
    element : dict

    """
    def __init__(self, mesh_file):
        self.nodes, self.elements = parse_msh(mesh_file)
        self.name = mesh_file.split(".", 1)[0]
