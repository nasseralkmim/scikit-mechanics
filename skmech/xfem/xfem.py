"""Construct an object with xfem parameters
"""
import numpy as np
from .distance import distance


class Xfem(object):
    """Model for xfem parameters

    Parameters
    ----------
    zerolevelset: object
        object with grid_x, grid_y and mask attributes created with
        skmech.xfem.zerolevelset.ZeroLevelSet() class.

    Attributes
    ----------
    enr_nodes : list
        used in gradient matrix and to numerate new enriched dofs
    enr_elements : ndarray shape
        used to decide if an element is enriched or not
    zls : dict
        dictionary containing the zero level set object and its attributes,
        namely: phi, enriched_nodes and enriched elements

    """
    def __init__(self, nodes, elements, zerolevelset, material):
        self.xfem = True
        self.nodes = nodes
        self.elements = elements
        self.material = material
        self.num_dof = len(nodes) * 2

        # if element is in matrix or reinforcement area
        self.element_material = {'matrix': [], 'reinforcement': []}

        # extract nodes coordinates for 2D as array
        xyz = np.array(list(nodes[n][:2] for n in nodes.keys()))

        self.enr_elements = []
        self.enr_nodes = []
        self.zls = {}
        # put zerolevelset in a list if only one object was given

        if not isinstance(zerolevelset, list):
            zerolevelset = [zerolevelset]

        for zid, zls_obj in enumerate(zerolevelset):
            # add the zerolevel objecto to a dictionary
            self.zls[zid] = zls_obj
            self.zls[zid].phi = distance(zls_obj.mask, zls_obj.grid_x,
                                         zls_obj.grid_y, xyz)
            self.zls[zid].enr_nodes = self._get_enriched_nodes(
                self.zls[zid].phi)
            self.zls[zid].enr_elements = self._get_enriched_elements(
                self.zls[zid].enr_nodes)
            # TODO: dof for each zls and then dof for element
            self.zls[zid].enr_node_dof = self._generate_enriched_dof(
                self.zls[zid].enr_nodes)

            self.enr_elements.extend(self.zls[zid].enr_elements)
            self.enr_nodes.extend(self.zls[zid].enr_nodes)

        # Important to decide if an ele is enr or not
        self.enr_elements = list(set(self.enr_elements))
        # Important to define new degree's of freedom numbering
        self.enr_nodes = list(set(self.enr_nodes))

    def _generate_enriched_dof(self, enr_nodes):
        """get enriched dofs for each zero level set"""
        enr_node_dof = {}
        for ind, enr_node_id in enumerate(enr_nodes):
            enr_node_dof[enr_node_id] = [ind*2 + self.num_dof + 1,
                                         ind*2 + self.num_dof + 1 + 1]
        self.num_dof += len(enr_nodes)*2
        return enr_node_dof

    def _get_enriched_nodes(self, phi):
        """Find the enriched nodes based on discontinuity elements"""
        discontinuity_elements = self._get_discontinuity_elements(phi)
        # find the enriched nodes associated with this zero level set
        # unordered
        enr_nodes = []
        for eid in discontinuity_elements:
            _, _, _, _, *conn = self.elements[eid]
            enr_nodes.extend(conn)
        return list(sorted(set(enr_nodes)))

    def _get_discontinuity_elements(self, phi):
        """Get elements that  have the discontinuity within them"""
        discontinuity_elements = []
        for eid, [_, _, _, _, *conn] in self.elements.items():
            conn = np.array(conn) - 1  # python starts at 0
            if np.all(phi[conn] < 0):
                # add element to reinforcement list
                self.element_material['reinforcement'].append(eid)
            elif np.all(phi[conn] > 0):
                self.element_material['matrix'].append(eid)
            else:
                discontinuity_elements.append(eid)
        return discontinuity_elements

    def _get_enriched_elements(self, enr_nodes):
        """Get enriched elements based on enriched nodes

        Note
        ----
        Includes blendind elements

        """
        enr_elements = []
        for eid, [_, _, _, _, *conn] in self.elements.items():
            if np.any(np.in1d(enr_nodes, conn)):
                enr_elements.append(eid)
        return enr_elements
