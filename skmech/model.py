"""Module for building the structure model

This module contains the class that creates the
structure model.
The structure model has all attributes from the mesh object.

"""
import numpy as np
import levelset
from .xfem.zerolevelset import Create


class Model(object):
    """Build the model object

    Parameters
    ----------
    mesh : object
        object containing mesh data from gmsh nodes and elements
    zerolevelset (optional): list with objects with zero level set
        definition created with the class:
            elastopy.xfem.zerolevelset.Create()
        each object has the mask array that defines the interface, the mask
        consists of -1 and 1, the interface is where the sign changes.

    Attributes
    ----------
    mesh attributes from elastopy.mesh.gmsh.Parse() object
    enriched_elements (list): elements that are enriched used in
        constructing elements
    enriched_nodes (list): list with all nodes enriched (global tag)
        sorted.
    num_enr_dof (int): number of enriched dofs for whole model
    num_std_dof (int): number of standard dofs for whole model
    xfem (boolean): indicates if xfem is used
    phi (numpy array): shape (num_nodes, ) signed distance function from
        the zero level set contour to mesh nodes.
    material (obj): aggregation
    zerolevelset (list of obj default []): aggregation of list
    thickness (float default 1.0): thickness of model in meters (SI)

    Note
    ----
    The enrichment attributes are set if the zerolevelset is passed
    as a non None argument.

    Note
    ----
    Updates the number of dofs and the element dofs if the zero
    level set function is set, which means that enrichments will
    be used.
    The enriched dofs are set by continuing counting from the last
    global dof tag.
    The order is defined by enriched nodes list which comes resulted
    from the built in set function.

    Example:
        # standard dofs with global tags
        DOF = [[0, 1, 2, 3, 4, 5, 6, 7],
               [2, 3, 8, 9, 14, 15, 4, 5],
               [8, 9, 10, 11, 12, 13, 14, 15]]
        enriched_nodes = [1, 2, 4, 7]
        enriched_DOF = [16, 17, 18, 19, 20, 21, 22, 23]

    where, 16 = 2*(0) + 16 and 21 = 2*(2) + 16 + {1}, the number in
    parenthesis is the index of the node tag in the enriched_nodes list
    and the number in {} is the additional dof for the 2d mechanics
    problem.

    Note
    ----
    Each zerolevelset entry will have new attributes created:
        1. signed distance function (phi) (array shape (num_nodes, ))
        2. discontinuity_elements (list): list with elements that are cut
            by discontinuity interface defined by the zero level set
        3. enriched_nodes (list): list of nodes marked for enrichment
        4. enriched_elements (list): list of enriched elements for this zls
        5. enriched dof (dictionary): enriched dof associated to each
            enriched element, this is sorted.

    """
    def __init__(self, mesh,
                 material=None,
                 traction=None,
                 displacement=None,
                 body_forces=None,
                 zerolevelset=[],
                 num_quad_points=4, thickness=1., etypes=[3]):
        self.mesh = mesh        # aggregation
        self.traction = traction
        self.body_forces = body_forces
        self.displacement = displacement
        self.elements = self._get_elements(mesh.elements, etypes)
        self.thickness = thickness
        # only for surface material, levelsets
        if material is not None:
            self.material = material  # aggregate material object to model

        self.num_ele = len(self.mesh.elements)
        self.num_quad_points = self._get_number_quad_points(num_quad_points)
        self.num_nodes = len(self.mesh.nodes)
        self.num_dof_node = self._get_number_dof(etypes)
        self.num_dof = self.num_nodes * self.num_dof_node
        self.enriched_elements = []
        self.nodes_dof = self._generate_dof()

    def get_physical_element(self, physical_element):
        """get physical line element

        Parameters
        ----------
        physical_element : int
            tag of the physical element where the traction is assigned
        elements : dict
            dictionary with all mesh elements from gmsh

        Returns
        -------
        dict
            dictionary with element data that are in the physical element

        """
        return {key: value
                for key, value in self.mesh.elements.items()
                if value[2] == physical_element}

    def _generate_dof(self):
        """Generate nodal degree of freedom

        This is done based on nodal tag, element type and number of dof
        per node.

        Returns
        -------
        dict
            dictionary with node tag and respective degree's of freedom
            associated with this node

        """
        dof = {}
        for node_id, _ in self.mesh.nodes.items():
            dof[node_id] = [node_id * self.num_dof_node - 1 + i
                            for i in range(self.num_dof_node)]
        return dof

    def _get_elements(self, elements, etypes):
        """Select only declared elements from msh file

        Parameters
        ----------
        elements : dict
            dictionary with all elements produced by gmsh
        etypes : list
            list with element types

        Returns
        -------
        dict
            dictionary with element type and element data if element
            type is in declared etypes list

        """
        return {key: value
                for key, value in elements.items()
                if value[0] in etypes}

    def _get_number_dof(self, etypes):
        """Compute number os dof per node"""
        if 3 in etypes:
            num_dof_node = 2
        elif 1 in etypes:
            # spatial frame
            num_dof_node = 6
        elif 5 in etypes:
            # hexaedro
            num_dof_node = 3
        return num_dof_node

    def _get_number_quad_points(self, num_quad_points):
        """Compute number of quad points for each element

        Parameters
        ----------
        num_quad_points : int

        Returns
        -------
        dict
            dictionary with element id and number of quad points

        """
        return {key: value
                for key, value in
                zip(self.mesh.elements.keys(),
                    [num_quad_points]*self.num_ele)}

        # # check if zerolevelset is defined
        # self.xfem = False
        # # empty list is false
        # if zerolevelset:
        #     self.xfem = True
        #     # put it in a list of len=1 if only one level set is given
        #     if type(zerolevelset) is not list:
        #         zerolevelset = [zerolevelset]
        
        # self.enriched_nodes = []

        # self.num_enr_dof = 0
        # # list with zerolevelset objects
        # self.zerolevelset = []
        # # loop over zerolevelset objects
        # for zls in zerolevelset:
        #     # TODO: better way to set material properties
        #     if hasattr(zls, 'material'):
        #         self.E_matrix = zls.material.E[1]
        #         self.nu_matrix = zls.material.nu[1]
        #     # update the max dof id when DOF includes enriched dof for a zls
        #     # +1 to start the count enr dofs
        #     max_dof_id = max(max(dof) for dof in self.DOF) + 1

        #     # discontinuity elements of this level set
        #     zls.discontinuity_elements = []
        #     zls.enriched_nodes = []
        #     zls.enriched_elements = []
        #     zls.enriched_dof = {}
        #     # zls.phi shape (num_nodes, ) with signed distance value
        #     zls.phi = levelset.distance(zls.mask_ls,
        #                                 zls.grid_x,
        #                                 zls.grid_y,
        #                                 self.XYZ)

        #     for e, conn in enumerate(self.CONN):
        #         # check if element is enriched or not
        #         if np.all(zls.phi[conn] < 0) or np.all(
        #                 zls.phi[conn] > 0):
        #             pass
        #         else:
        #             zls.discontinuity_elements.append(e)

        #     # find the enriched nodes associated with this zero level set
        #     # unordered
        #     for e in zls.discontinuity_elements:
        #         zls.enriched_nodes.extend(self.CONN[e])

        #     # set extracts an unordered collection of unique elements
        #     # model enriched nodes list is sorted!
        #     zls.enriched_nodes = list(sorted(set(zls.enriched_nodes)))
        #     self.enriched_nodes.extend(zls.enriched_nodes)

        #     # mark enriched and blending elements for this zero level set
        #     for e, conn in enumerate(self.CONN):
        #         # check if any enriched node is in conn
        #         if np.any(np.in1d(zls.enriched_nodes, conn)):
        #             zls.enriched_elements.append(e)

        #             # find the enriched dofs
        #             # loop over enriched nodes in element
        #             dof = []
        #             for n in np.intersect1d(zls.enriched_nodes, conn):
        #                 # intersect1d returns sorted nodes
        #                 # find the index of the node in the enriched nodes
        #                 ind = np.where(zls.enriched_nodes == n)[0][0]
        #                 dof.append(ind*2 + max_dof_id)
        #                 dof.append(ind*2 + max_dof_id + 1)
        #                 # global dof numbering for element
        #                 self.DOF[e].append(ind*2 + max_dof_id)
        #                 self.DOF[e].append(ind*2 + max_dof_id + 1)
        #             # enriched dof for element e in this zero level set
        #             zls.enriched_dof[e] = dof

        #     # add 2 new dofs for each enriched node for each level set
        #     self.num_dof += 2*len(zls.enriched_nodes)
        #     # total number of enriched dofs for whole model
        #     self.num_enr_dof += 2*len(zls.enriched_nodes)

        #     # append the zero level set to model
        #     self.zerolevelset.append(zls)
        #     # list of all enriched elements
        #     self.enriched_elements.extend(zls.enriched_elements)

        # self.num_std_dof = self.num_dof - self.num_enr_dof
        # self.enriched_elements = list(set(self.enriched_elements))  # Notsorted
        # self.enriched_nodes = list(sorted(set(self.enriched_nodes)))


if __name__ == '__main__':
    class Mesh():
        pass
    mesh = Mesh()
    mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    mesh.CONN = np.array([[0, 1, 2, 3]])
    mesh.num_ele = 1
    mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7]]
    mesh.num_dof = 8

    # center in (1, 1) and radius .2
    def func(x, y):
        # return (x - 1)**2 + (y - 1)**2 - 0.2**2
        return x - .5

    z_ls = Create(func, [0, 1], [0, 1], num_div=3)

    model = Build(mesh, zerolevelset=[z_ls])

