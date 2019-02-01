"""Builde micromodel with mesh and etc"""
import numpy as np
from ..model import Model
from ..mesh.mesh import Mesh


class MicroModel(Model):
    """Build the microscal self based on the self of macroscale

    Note
    ----
    The main difference is that in the microscale we need an coordinate
    matrix C, and the indexex of boundary, internal nodes dofs

    """
    def __init__(self, mesh=None, material=None, traction=None,
                 displacement_bc=None, body_forces=None, zerolevelset=None,
                 imposed_displ=None, num_quad_points=2, thickness=1.,
                 etypes=[3], macro_element_out=None, macro_gpid_out=None):

        # call the parent class to build a fem self
        # just for testing porpuses, this should be specified in the problem
        # description
        if mesh is None:
            mesh = Mesh('microscale/microdomain.msh')

        # specify conditions to get all dofs on the boundary and interior
        displacement_bc = {1: (None, None), 2: (None, None),
                           3: (None, None), 4: (None, None)}
        super(MicroModel, self).__init__(mesh=mesh, material=material,
                                         traction=traction,
                                         displacement_bc=displacement_bc,
                                         body_forces=body_forces,
                                         zerolevelset=zerolevelset,
                                         imposed_displ=imposed_displ,
                                         num_quad_points=num_quad_points,
                                         thickness=thickness, etypes=etypes)
        # boundary and internal nodes id and dofs id
        self.nid_b, self.nid_i = self.get_boundary_internal_nodes()
        self.id_b, self.id_i = self.get_boundary_internal_dof()
        # build coordinate matrix
        self.C_b, self.C_i, self.C_g = self._get_global_coordinate_matrix()

        self.area = self._get_microdomain_area()
        self.volume = self.area * self.thickness

        # initial micro scale displacement
        # when micromodel is defined
        self.u_mic = np.zeros(self.num_dof)
        # use to store the Delta_u_fluct associated with the macro strain
        # increment Delta_eps_macro for each Meid, Mgpid
        # this is used to update micro displacement after macro NR converged 
        self.Delta_u_fluct = {}
        # macroscopic strain associated with each micro model
        self.Delta_eps_mac = {}

    def _get_microdomain_area(self):
        """Compute element area"""
        xmax = max([x for [x, _, _] in self.nodes.values()])
        xmin = min([x for [x, _, _] in self.nodes.values()])
        ymax = max([y for [_, y, _] in self.nodes.values()])
        ymin = min([y for [_, y, _] in self.nodes.values()])
        area = abs(xmax - xmin) * abs(ymax - ymin)
        return area

    def update_displ(self, element_out):
        """Update the micro displacement attribute

        Update with its previous value stored in the attribute u_mic,
        with the macro strain increment and increment in the fluct

        Parameters
        ----------
        element_out : int
            element for plotting micro model displacemen
        """
        displ_mac = np.zeros(self.num_dof)
        # add displacement due macro strain increment
        # the incidence index should match the coordinate matrix C
        Delta_eps_mac = self.Delta_eps_mac[(element_out, 3)][:3]
        displ_mac[self.id_i + self.id_b] = self.C_g.T @ Delta_eps_mac
        u_mic = displ_mac + self.Delta_u_fluct[(element_out, 3)]
        return u_mic

    def get_boundary_internal_dof(self):
        """Get internal and boundary nodes

        Note
        ----
        dof follow the nid order

        """
        id_b, id_i = [], []
        for nid_b in self.nid_b:
            # subtract one to start at 0 for dof numbering
            dof = list(np.array(self.nodes_dof[nid_b]) - 1)
            id_b.extend(dof)
        for nid_i in self.nid_i:
            dof = list(np.array(self.nodes_dof[nid_i]) - 1)
            id_i.extend(dof)
        return id_b, id_i

    def _get_coordinate_matrix(self, nid):
        """Assemble the coordinate matrix for a node"""
        x, y = self.nodes[nid][0], self.nodes[nid][1]
        C = np.array([[x, 0],
                      [0, y],
                      [y / 2, x / 2]])
        return C

    def _get_global_coordinate_matrix(self):
        """Assemble global coordinate matrix

        b for boundary nodes, i for internal nodes and g for global

        Note
        ----
        The order of the matrix follows the node order of the nid_i, nid_b
        lists
        """
        Cb, Ci = {}, {}
        for nid in self.nodes.keys():
            if nid in self.nid_b:
                Cb[nid] = self._get_coordinate_matrix(nid)
            if nid in self.nid_i:
                Ci[nid] = self._get_coordinate_matrix(nid)
        C_b = np.block([Cb[nid] for nid in self.nid_b])
        C_i = np.block([Ci[nid] for nid in self.nid_i])
        C_g = np.block([C_i, C_b])
        return C_b, C_i, C_g

    def get_boundary_internal_nodes(self):
        """Create a list with free and restrained nodes

        Note
        ----
        Used the items in the displacement_bc dictionary

        """
        nid_b = []
        if self.displacement_bc is not None:
            for d_loc, d_value in self.displacement_bc.items():
                physical_element = self.get_physical_element(d_loc)
                for eid, [etype, *edata] in physical_element.items():
                    # physical points, Not used for this application
                    if etype == 15:
                        node = edata[-1]  # last entry
                        nid_b.append(node)
                    # physical lines
                    if etype == 1:
                        node_1, node_2 = edata[-2], edata[-1]
                        nid_b.append(node_1)
                        nid_b.append(node_2)

        nid_i = [n for n in self.nodes.keys()
                 if n not in nid_b]
        # set for getting only unique dofs
        return list(set(nid_b)), nid_i

    def set_internal_var(self, int_var, macromodel):
        """Set up internal variable into the int_var dict

        Note
        ----
        This data structure seems to be very complex and not optimal
        it is a prototype

        int_var dict with internal variables: eps, eps_e, eps_p ...
            (Meid, Mgpid) each internal variable has a dict with an element and gp key
                (eid, gpid) each Mgp will have another dict with micro element gp key
        
        """
        # add into internal variable dictionary the microscale variables
        eps_e = {(Meid, Mgpid): {(eid, gpid): np.zeros(4)
                                 for eid in self.elements.keys()
                                 for gpid in range(self.num_quad_points[eid] * 2)}
                 for Meid in macromodel.elements.keys()
                 for Mgpid in range(macromodel.num_quad_points[Meid] * 2)}
        eps_p = {(Meid, Mgpid): {(eid, gpid): np.zeros(4)
                                 for eid in self.elements.keys()
                                 for gpid in range(self.num_quad_points[eid] * 2)}
                 for Meid in macromodel.elements.keys()
                 for Mgpid in range(macromodel.num_quad_points[Meid] * 2)}
        eps_bar_p = {(Meid, Mgpid): {(eid, gpid): 0
                                     for eid in self.elements.keys()
                                     for gpid in range(self.num_quad_points[eid] * 2)}
                    for Meid in macromodel.elements.keys()
                    for Mgpid in range(macromodel.num_quad_points[Meid] * 2)}
        dgamma = {(Meid, Mgpid): {(eid, gpid): 0
                                  for eid in self.elements.keys()
                                  for gpid in range(self.num_quad_points[eid] * 2)}
                  for Meid in macromodel.elements.keys()
                  for Mgpid in range(macromodel.num_quad_points[Meid] * 2)}

        # copy beacause we need them in the first step, and they start with 0
        int_var['eps_e_mic'] = eps_e.copy()
        int_var['eps_p_mic'] = eps_p.copy()
        int_var['eps_bar_p_mic'] = eps_bar_p.copy()
        int_var['dgamma_mic'] = dgamma.copy()
        int_var['sig_mic'] = {}
        int_var['eps_mic'] = {}
        int_var['q_mic'] = {}
        return int_var


if __name__ == '__main__':
    micromodel = MicroModel()
