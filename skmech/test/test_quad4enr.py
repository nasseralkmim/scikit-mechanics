import numpy as np
from elastopy import gmsh, Build, Material, boundary
from elastopy.constructor import constructor
from elastopy import xfem
from elastopy.xfem.zerolevelset import Create

np.set_printoptions(precision=2, suppress=True)


def test_1zls():
    def b_force(x1, x2, t=1):
        return np.array([0.0, 0.0])

    def trac_bc(x1, x2, t=1):
        return {('line', 1): [2e7, 0]}  # kg/m

    def displ_bc(x1, x2):
        return {('node', 0): [0, 0], ('node', 3): [0, 0]}

    EPS0 = None
    func = lambda x, y: x - 0.3

    mesh = gmsh.Parse('examples/xfem_bimaterial')
    material = Material(E={-1: 2e11, 1: 1e11},
                        nu={-1: .3, 1: .3},
                        case='strain')  # kg/m2
    zls = xfem.zerolevelset.Create(func, [0, .6], [0, .2], num_div=100,
                                   material=material)
    model = Build(mesh, zerolevelset=zls, thickness=.01)
    K = np.zeros((model.num_dof, model.num_dof))
    k = {}
    for eid, type in enumerate(model.TYPE):
        element = constructor(eid, model, EPS0)
        k[eid] = element.stiffness_matrix()
        K[element.id_m] += k[eid]

    kstdele0 = np.array(
        [[1.154, 0.481, -0.769, 0.096, -0.577, -0.481, 0.192, -0.096],
         [0.481, 1.154, -0.096, 0.192, -0.481, -0.577, 0.096, -0.769],
         [-0.769, -0.096, 1.154, -0.481, 0.192, 0.096, -0.577, 0.481],
         [0.096, 0.192, -0.481, 1.154, -0.096, -0.769, 0.481, -0.577],
         [-0.577, -0.481, 0.192, -0.096, 1.154, 0.481, -0.769, 0.096],
         [-0.481, -0.577, 0.096, -0.769, 0.481, 1.154, -0.096, 0.192],
         [0.192, 0.096, -0.577, 0.481, -0.769, -0.096, 1.154, -0.481],
         [-0.096, -0.769, 0.481, -0.577, 0.096, 0.192, -0.481, 1.154]])
    assert np.allclose(k[0][:8, :8] / 1e9, kstdele0, atol=1e-2)
    kenrele0 = np.array(
        [[129.915, 0., 49.573, -0.], [-0., 70.085, -0., -18.803],
         [49.573, -0., 129.915, 0.], [0., -18.803, 0., 70.085]])
    assert np.allclose(k[0][8:, 8:]/1e9*3600, kenrele0, atol=1e-2)


def test_2zls():
    """Plate with 2 zls, left with E=2e11 and right E=5e11
    2 elements
    check how two or more level sets are dealt
    """

    class Mesh():
        pass

    mesh = Mesh()
    mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 1], [2, 0]])
    mesh.CONN = np.array([[0, 1, 2, 3], [1, 5, 4, 2]])
    mesh.num_ele = 2
    mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 10, 11, 8, 9, 4, 5]]
    mesh.num_dof = 12
    mesh.TYPE = [3, 3]
    mesh.surf_of_ele = [0, 0]
    mesh.bound_ele = np.array([[0, 0, 0], [0, 2, 2], [0, 3, 3], [1, 0, 0],
                               [1, 1, 1], [1, 2, 2]])
    mesh.nodes_in_bound_line = np.array([[0, 0, 1], [0, 1, 4], [1, 4, 5],
                                         [2, 5, 2], [2, 2, 3], [3, 3, 0]])
    fun1 = lambda x, y: x - .5
    fun2 = lambda x, y: (-x + 1.5)
    mat_zls0 = Material(E={-1: 2e11, 1: 1e11},
                        nu={-1: .3, 1: .3},
                        case='strain')
    mat_zls1 = Material(E={-1: 5e11, 1: 1e11},
                        nu={-1: .3, 1: .3},
                        case='strain')

    zls = [Create(fun1, [0, 4], [0, 1], material=mat_zls0, num_div=50),
           Create(fun2, [0, 4], [0, 1], material=mat_zls1, num_div=50)]

    model = Build(mesh, zerolevelset=zls, thickness=.01,
                  num_quad_points=2)
    # enriched nodes are sorted due np.intersect1d
    assert model.zerolevelset[0].enriched_nodes == [0, 1, 2, 3]
    assert model.zerolevelset[1].enriched_nodes == [1, 2, 4, 5]

    EPS0 = None
    K = np.zeros((model.num_dof, model.num_dof))
    k = {}
    element = {}
    for eid, type in enumerate(model.TYPE):
        element[eid] = constructor(eid, model, EPS0)

        k[eid] = element[eid].stiffness_matrix()
        K[element[eid].id_m] += k[eid]

    # 3x8 standard 3x8 due one level set and 3x4 due the other
    assert np.shape(k[0]) == (20, 20)
    print(element[0].E)
    print(element[1].E)
    assert np.allclose(element[0].E, np.array([219780219780.2198,
                                               109890109890.1099,
                                               109890109890.1099,
                                               219780219780.2198]),
                       rtol=1e-2, atol=1e-2)
    assert np.allclose(element[1].E, np.array([109890109890.1099,
                                               549450549450.54944,
                                               549450549450.54944,
                                               109890109890.1099]),
                       rtol=1e-2, atol=1e-2)
    assert element[1].nu == [0.4285714285714286,
                             0.4285714285714286,
                             0.4285714285714286,
                             0.4285714285714286]
    assert model.zerolevelset[0].enriched_elements == [0, 1]
    assert model.zerolevelset[1].enriched_elements == [0, 1]
    assert len(element[0].zerolevelset) == 2
    assert len(element[1].zerolevelset) == 2
    assert element[0].dof == [0, 1, 2, 3, 4, 5, 6, 7,  # standard dof
                              12, 13, 14, 15, 16, 17, 18, 19,  # enr due zls1
                              20, 21, 22, 23]

    # standard dof follow connective order
    # enriched dof follow element.enriched_nodes order
    # element enriched nodes is sorted
    assert element[1].dof == [2, 3, 10, 11, 8, 9, 4, 5,  # std
                              14, 15, 16, 17,  # first zls
                              20, 21, 22, 23, 24, 25, 26, 27]  # second zls
    assert model.zerolevelset[0].enriched_dof[1] == [14, 15, 16, 17]
    assert model.zerolevelset[1].enriched_dof[1] == [20, 21, 22, 23,
                                                     24, 25, 26, 27]
    assert list(element[1].enriched_nodes[0]) == [1, 2]
    assert list(element[1].enriched_nodes[1]) == [1, 2, 4, 5]  # sorted
    assert model.zerolevelset[0].enriched_nodes == [0, 1, 2, 3]
    assert model.zerolevelset[1].enriched_nodes == [1, 2, 4, 5]


def test_4zls():
    class Mesh():
        pass
    mesh = Mesh()
    mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
                         [2, 0], [2, 1],
                         [3, 0], [3, 1],
                         [4, 0], [4, 1]])
    mesh.CONN = np.array([[0, 1, 2, 3],
                          [1, 4, 5, 2],
                          [4, 6, 7, 5],
                          [6, 8, 9, 7]])
    mesh.num_ele = 4
    mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7],
                [2, 3, 8, 9, 10, 11, 4, 5],
                [8, 9, 12, 13, 14, 15, 10, 11],
                [12, 13, 16, 17, 18, 19, 14, 15]]
    mesh.num_dof = 20
    mesh.TYPE = [3, 3, 3, 3]
    mesh.surf_of_ele = [0, 0, 0, 0]
    # these next two arrays are incorret but are irrelevant for this test
    mesh.bound_ele = np.array([[0, 0, 0], [0, 2, 2], [0, 3, 3], [1, 0, 0],
                               [1, 1, 1], [1, 2, 2]])
    mesh.nodes_in_bound_line = np.array([[0, 0, 1], [0, 1, 4], [1, 4, 5],
                                         [2, 5, 2], [2, 2, 3], [3, 3, 0]])

    fun1 = lambda x, y: x - .5
    fun2 = lambda x, y: (-x + 2.5)
    mat_zls0 = Material(E={-1: 2e11, 1: 1e11},
                        nu={-1: .3, 1: .3},
                        case='stress')
    mat_zls1 = Material(E={-1: 5e11, 1: 1e11},
                        nu={-1: .3, 1: .3},
                        case='stress')
    zls = [Create(fun1, [0, 4], [0, 1], material=mat_zls0, num_div=50),
           Create(fun2, [0, 4], [0, 1], material=mat_zls1, num_div=50)]

    model = Build(mesh, zerolevelset=zls)

    EPS0 = None
    k = {}
    element = {}
    for eid, type in enumerate(model.TYPE):
        element[eid] = constructor(eid, model, EPS0)
        k[eid] = element[eid].stiffness_matrix()
        print(element[eid].E)
    assert element[0].E == [2e11, 1e11, 1e11, 2e11]
    assert element[1].E == [1e11]*4
    assert element[2].E == [1e11, 5e11, 5e11, 1e11]
    assert element[3].E == [5e11]*4

