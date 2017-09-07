"""Test quad enr element

Tests:

1. Element material properties
2. Stiffness matrix

"""
import numpy as np
import skmech


def test_element():
    """test element material"""
    class Mesh():
        def __init__(self):
            self.nodes = {
                1: [0, 0, 0],
                2: [.2, 0, 0],
                3: [.2, .2, 0],
                4: [0, 0.2, 0],
                5: [.4, 0, 0],
                6: [.6, 0, 0],
                7: [.6, .2, 0],
                8: [.4, .2, 0],
            }
            self.elements = {
                1: [15, 2, 12, 1, 1],
                2: [15, 2, 13, 4, 4],
                13: [1, 2, 3, 30, 6, 7],
                3: [1, 2, 7, 2, 3, 7],
                4: [1, 2, 7, 2, 7, 4],
                9: [3, 2, 11, 10, 1, 2, 3, 4],
                10: [3, 2, 11, 10, 2, 5, 8, 3],
                11: [3, 2, 11, 10, 5, 6, 7, 8],
            }

    msh = Mesh()
    func = lambda x, y: x - 0.3
    zls = skmech.xfem.ZeroLevelSet(func, [0, .6], [0, .2], num_div=100)
    mat = skmech.Material(E={-1: 2e11, 1: 1e11}, nu={-1: 0.3, 1: 0.3},
                          case='strain')
    model = skmech.Model(msh, zerolevelset=zls, material=mat,
                         thickness=0.01)

    k = {}
    emat = {}
    for eid, [etype, *_] in model.elements.items():
        ele = skmech.constructor(eid, etype, model)
        emat[eid] = ele.E

    assert emat[11] == [109890109890.1099,
                        109890109890.1099,
                        109890109890.1099,
                        109890109890.1099]
    assert emat[10] == [219780219780.2198,
                        109890109890.1099,
                        109890109890.1099,
                        219780219780.2198]


def test_element_stiffness():
    """Test enriched element stiffness matrix"""
    k_std_ele0 = np.array(
        [[1.154, 0.481, -0.769, 0.096, -0.577, -0.481, 0.192, -0.096],
         [0.481, 1.154, -0.096, 0.192, -0.481, -0.577, 0.096, -0.769],
         [-0.769, -0.096, 1.154, -0.481, 0.192, 0.096, -0.577, 0.481],
         [0.096, 0.192, -0.481, 1.154, -0.096, -0.769, 0.481, -0.577],
         [-0.577, -0.481, 0.192, -0.096, 1.154, 0.481, -0.769, 0.096],
         [-0.481, -0.577, 0.096, -0.769, 0.481, 1.154, -0.096, 0.192],
         [0.192, 0.096, -0.577, 0.481, -0.769, -0.096, 1.154, -0.481],
         [-0.096, -0.769, 0.481, -0.577, 0.096, 0.192, -0.481, 1.154]])
    # assert np.allclose(k[0][:8, :8] / 1e9, kstdele0, atol=1e-2)
    k_enr_ele0 = np.array(
        [[129.915, 0., 49.573, -0.], [-0., 70.085, -0., -18.803],
         [49.573, -0., 129.915, 0.], [0., -18.803, 0., 70.085]])


def test_dof():
    """test dof for 2 zero level set thtat don't share an element"""
    class Mesh():
        def __init__(self):
            self.nodes = {
                1: [0, 0, 0],
                2: [1, 0, 0],
                3: [1, 1, 0],
                4: [0, 1, 0],
                5: [2, 0, 0],
                6: [2, 1, 0],
                7: [3, 0, 0],
                8: [3, 1, 0],
                9: [4, 0, 0],
                10: [4, 1, 0]
            }
            self.elements = {
                1: [3, 2, 11, 10, 1, 2, 3, 4],
                2: [3, 2, 11, 10, 2, 5, 6, 3],
                3: [3, 2, 11, 10, 5, 7, 8, 6],
                4: [3, 2, 11, 10, 7, 9, 10, 8]
            }
    msh = Mesh()
    func = lambda x, y: x - 0.5
    func2 = lambda x, y: (-x + 3.5)
    zls = [skmech.xfem.ZeroLevelSet(func, [0, 4], [0, 1], num_div=50),
           skmech.xfem.ZeroLevelSet(func2, [0, 4], [0, 1], num_div=50)]
    mat = skmech.Material(E={-1: 2e11, 1: 1e11}, nu={-1: 0.3, 1: 0.3},
                          case='strain')
    model = skmech.Model(msh, zerolevelset=zls, material=mat,
                         thickness=0.01)
    assert model.xfem.enr_elements == [1, 2, 3, 4]

    dof = {}
    for eid, [etype, *_] in model.elements.items():
        ele = skmech.constructor(eid, etype, model)
        dof[eid] = ele.dof
    assert dof[1] == [1, 2, 3, 4, 5, 6, 7, 8,
                      21, 22, 23, 24, 25, 26, 27, 28]
    assert dof[4] == [13, 14, 17, 18, 19, 20, 15, 16,
                      29, 30, 31, 32, 33, 34, 35, 36]


def test_dof2():
    """test dof for two zero level sets"""
    class Mesh():
        def __init__(self):
            self.nodes = {
                1: [0, 0, 0],
                2: [1, 0, 0],
                3: [1, 1, 0],
                4: [0, 1, 0],
                5: [2, 0, 0],
                6: [2, 1, 0],
            }
            self.elements = {
                1: [3, 2, 11, 10, 1, 2, 3, 4],
                2: [3, 2, 11, 10, 2, 5, 6, 3],
            }
    msh = Mesh()
    func = lambda x, y: x - 0.5
    func2 = lambda x, y: (-x + 1.5)
    zls = [skmech.xfem.ZeroLevelSet(func, [0, 2], [0, 1], num_div=50),
           skmech.xfem.ZeroLevelSet(func2, [0, 2], [0, 1], num_div=50)]
    mat = skmech.Material(E={-1: 2e11, 1: 1e11}, nu={-1: 0.3, 1: 0.3},
                          case='strain')
    model = skmech.Model(msh, zerolevelset=zls, material=mat,
                         thickness=0.01)
    assert model.xfem.enr_elements == [1, 2]
    assert model.xfem.enr_nodes == [1, 2, 3, 4, 5, 6]

    assert model.xfem.zls[0].enr_nodes == [1, 2, 3, 4]
    assert model.xfem.zls[1].enr_nodes == [2, 3, 5, 6]
    assert model.xfem.zls[0].enr_node_dof == {1: [13, 14], 2: [15, 16],
                                              3: [17, 18], 4: [19, 20]}
    assert model.xfem.zls[1].enr_node_dof == {2: [21, 22], 3: [23, 24],
                                              5: [25, 26], 6: [27, 28]}

    dof = {}
    for eid, [etype, *_] in model.elements.items():
        ele = skmech.constructor(eid, etype, model)
        dof[eid] = ele.dof
        print(ele.dof)

    # element dof numbering following CCW for standards and following
    # the sorted enr node order for each zero level set
    assert dof[1] == [1, 2, 3, 4, 5, 6, 7, 8,
                      13, 14, 15, 16, 17, 18, 19, 20,  # first zls
                      21, 22, 23, 24]                  # second
    assert dof[2] == [3, 4, 9, 10, 11, 12, 5, 6,
                      15, 16, 17, 18,  # node 2, 3 enr dof for first zls
                      21, 22, 23, 24, 25, 26, 27, 28]  # nodes 2,3,5,6 2nd zls


def test_dof3():
    """test dof for 4 zero level sets in separate 4 elements"""
    class Mesh():
        def __init__(self):
            self.nodes = {
                1: [0, 0, 0],
                2: [1, 0, 0],
                3: [1, 1, 0],
                4: [0, 1, 0],
                5: [2, 0, 0],
                6: [2, 1, 0],
                7: [2, 2, 0],
                8: [1, 2, 0],
                9: [0, 2, 0]
            }
            self.elements = {
                1: [3, 2, 11, 10, 1, 2, 3, 4],
                2: [3, 2, 11, 10, 2, 5, 6, 3],
                3: [3, 2, 11, 10, 3, 6, 7, 8],
                4: [3, 2, 11, 10, 4, 3, 8, 9]
            }
    msh = Mesh()
    func1 = lambda x, y: (x - 0)**2 + (y - 0)**2 - .2**2
    func2 = lambda x, y: (x - 2)**2 + (y - 0)**2 - .2**2
    func3 = lambda x, y: (x - 2)**2 + (y - 2)**2 - .2**2
    func4 = lambda x, y: (x - 0)**2 + (y - 2)**2 - .2**2
    zls = [skmech.xfem.ZeroLevelSet(func1, [0, 2], [0, 2], num_div=50),
           skmech.xfem.ZeroLevelSet(func2, [0, 2], [0, 2], num_div=50),
           skmech.xfem.ZeroLevelSet(func3, [0, 2], [0, 2], num_div=50),
           skmech.xfem.ZeroLevelSet(func4, [0, 2], [0, 2], num_div=50)]
    mat = skmech.Material(E={-1: 2e11, 1: 1e11}, nu={-1: 0.3, 1: 0.3},
                          case='stress')
    model = skmech.Model(msh, zerolevelset=zls, material=mat,
                         thickness=0.01)

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    skmech.xfem.plot2dzls(zls, ax)
    skmech.plot.geometry(msh.nodes, msh.elements, ax)
    plt.show()

    dof = {}
    for eid, [etype, *_] in model.elements.items():
        ele = skmech.constructor(eid, etype, model)
        dof[eid] = ele.dof

    assert dof[1] == [1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 21, 22, 23, 24, 25,
                      26, 27, 28, 29, 30, 35, 36, 43, 44, 45, 46]
    assert dof[2] == [3, 4, 9, 10, 11, 12, 5, 6, 21, 22, 23, 24, 27, 28, 29,
                      30, 31, 32, 33, 34, 35, 36, 37, 38, 43, 44]
    assert dof[3] == [5, 6, 11, 12, 13, 14, 15, 16, 23, 24, 29, 30, 33, 34, 35,
                      36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48]
    assert dof[4] == [7, 8, 5, 6, 15, 16, 17, 18, 23, 24, 25, 26, 29, 30, 35,
                      36, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]

test_dof3()













































# from elastopy import gmsh, Build, Material, boundary
# from elastopy.constructor import constructor
# from elastopy import xfem
# from elastopy.xfem.zerolevelset import Create

# np.set_printoptions(precision=2, suppress=True)


# def test_1zls():
#     def b_force(x1, x2, t=1):
#         return np.array([0.0, 0.0])

#     def trac_bc(x1, x2, t=1):
#         return {('line', 1): [2e7, 0]}  # kg/m

#     def displ_bc(x1, x2):
#         return {('node', 0): [0, 0], ('node', 3): [0, 0]}

#     EPS0 = None
#     FUNC = lambda x, y: x - 0.3

#     mesh = gmsh.Parse('examples/xfem_bimaterial')
#     material = Material(E={-1: 2e11, 1: 1e11},
#                         nu={-1: .3, 1: .3},
#                         case='strain')  # kg/m2
#     ZLS = xfem.zerolevelset.Create(FUNC, [0, .6], [0, .2], num_div=100,
#                                    material=material)
#     MODEL = Build(mesh, zerolevelset=ZLS, thickness=.01)
#     K = np.zeros((MODEL.num_dof, MODEL.num_dof))
#     k = {}
#     for eid, type in enumerate(MODEL.TYPE):
#         element = constructor(eid, MODEL, EPS0)
#         k[eid] = element.stiffness_matrix()
#         K[element.id_m] += k[eid]

#     kstdele0 = np.array(
#         [[1.154, 0.481, -0.769, 0.096, -0.577, -0.481, 0.192, -0.096],
#          [0.481, 1.154, -0.096, 0.192, -0.481, -0.577, 0.096, -0.769],
#          [-0.769, -0.096, 1.154, -0.481, 0.192, 0.096, -0.577, 0.481],
#          [0.096, 0.192, -0.481, 1.154, -0.096, -0.769, 0.481, -0.577],
#          [-0.577, -0.481, 0.192, -0.096, 1.154, 0.481, -0.769, 0.096],
#          [-0.481, -0.577, 0.096, -0.769, 0.481, 1.154, -0.096, 0.192],
#          [0.192, 0.096, -0.577, 0.481, -0.769, -0.096, 1.154, -0.481],
#          [-0.096, -0.769, 0.481, -0.577, 0.096, 0.192, -0.481, 1.154]])
#     assert np.allclose(k[0][:8, :8] / 1e9, kstdele0, atol=1e-2)
#     kenrele0 = np.array(
#         [[129.915, 0., 49.573, -0.], [-0., 70.085, -0., -18.803],
#          [49.573, -0., 129.915, 0.], [0., -18.803, 0., 70.085]])
#     assert np.allclose(k[0][8:, 8:]/1e9*3600, kenrele0, atol=1e-2)


# def test_2zls():
#     """Plate with 2 ZLS, left with E=2e11 and right E=5e11
#     2 elements
#     check how two or more level sets are dealt
#     """

#     class Mesh():
#         pass

#     mesh = Mesh()
#     mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1], [2, 1], [2, 0]])
#     mesh.CONN = np.array([[0, 1, 2, 3], [1, 5, 4, 2]])
#     mesh.num_ele = 2
#     mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7], [2, 3, 10, 11, 8, 9, 4, 5]]
#     mesh.num_dof = 12
#     mesh.TYPE = [3, 3]
#     mesh.surf_of_ele = [0, 0]
#     mesh.bound_ele = np.array([[0, 0, 0], [0, 2, 2], [0, 3, 3], [1, 0, 0],
#                                [1, 1, 1], [1, 2, 2]])
#     mesh.nodes_in_bound_line = np.array([[0, 0, 1], [0, 1, 4], [1, 4, 5],
#                                          [2, 5, 2], [2, 2, 3], [3, 3, 0]])
#     fun1 = lambda x, y: x - .5
#     fun2 = lambda x, y: (-x + 1.5)
#     mat_zls0 = Material(E={-1: 2e11, 1: 1e11},
#                         nu={-1: .3, 1: .3},
#                         case='strain')
#     mat_zls1 = Material(E={-1: 5e11, 1: 1e11},
#                         nu={-1: .3, 1: .3},
#                         case='strain')

#     ZLS = [Create(fun1, [0, 4], [0, 1], material=mat_zls0, num_div=50),
#            Create(fun2, [0, 4], [0, 1], material=mat_zls1, num_div=50)]

#     MODEL = Build(mesh, zerolevelset=ZLS, thickness=.01,
#                   num_quad_points=2)
#     # enriched nodes are sorted due np.intersect1d
#     assert MODEL.zerolevelset[0].enriched_nodes == [0, 1, 2, 3]
#     assert MODEL.zerolevelset[1].enriched_nodes == [1, 2, 4, 5]

#     EPS0 = None
#     K = np.zeros((MODEL.num_dof, MODEL.num_dof))
#     k = {}
#     element = {}
#     for eid, type in enumerate(MODEL.TYPE):
#         element[eid] = constructor(eid, MODEL, EPS0)

#         k[eid] = element[eid].stiffness_matrix()
#         K[element[eid].id_m] += k[eid]

#     # 3x8 standard 3x8 due one level set and 3x4 due the other
#     assert np.shape(k[0]) == (20, 20)
#     print(element[0].E)
#     print(element[1].E)
#     assert np.allclose(element[0].E, np.array([219780219780.2198,
#                                                109890109890.1099,
#                                                109890109890.1099,
#                                                219780219780.2198]),
#                        rtol=1e-2, atol=1e-2)
#     assert np.allclose(element[1].E, np.array([109890109890.1099,
#                                                549450549450.54944,
#                                                549450549450.54944,
#                                                109890109890.1099]),
#                        rtol=1e-2, atol=1e-2)
#     assert element[1].nu == [0.4285714285714286,
#                              0.4285714285714286,
#                              0.4285714285714286,
#                              0.4285714285714286]
#     assert MODEL.zerolevelset[0].enriched_elements == [0, 1]
#     assert MODEL.zerolevelset[1].enriched_elements == [0, 1]
#     assert len(element[0].zerolevelset) == 2
#     assert len(element[1].zerolevelset) == 2
#     assert element[0].dof == [0, 1, 2, 3, 4, 5, 6, 7,  # standard dof
#                               12, 13, 14, 15, 16, 17, 18, 19,  # enr due zls1
#                               20, 21, 22, 23]

#     # standard dof follow connective order
#     # enriched dof follow element.enriched_nodes order
#     # element enriched nodes is sorted
#     assert element[1].dof == [2, 3, 10, 11, 8, 9, 4, 5,  # std
#                               14, 15, 16, 17,  # first ZLS
#                               20, 21, 22, 23, 24, 25, 26, 27]  # second ZLS
#     assert MODEL.zerolevelset[0].enriched_dof[1] == [14, 15, 16, 17]
#     assert MODEL.zerolevelset[1].enriched_dof[1] == [20, 21, 22, 23,
#                                                      24, 25, 26, 27]
#     assert list(element[1].enriched_nodes[0]) == [1, 2]
#     assert list(element[1].enriched_nodes[1]) == [1, 2, 4, 5]  # sorted
#     assert MODEL.zerolevelset[0].enriched_nodes == [0, 1, 2, 3]
#     assert MODEL.zerolevelset[1].enriched_nodes == [1, 2, 4, 5]


# def test_4zls():
#     class Mesh():
#         pass
#     mesh = Mesh()
#     mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
#                          [2, 0], [2, 1],
#                          [3, 0], [3, 1],
#                          [4, 0], [4, 1]])
#     mesh.CONN = np.array([[0, 1, 2, 3],
#                           [1, 4, 5, 2],
#                           [4, 6, 7, 5],
#                           [6, 8, 9, 7]])
#     mesh.num_ele = 4
#     mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7],
#                 [2, 3, 8, 9, 10, 11, 4, 5],
#                 [8, 9, 12, 13, 14, 15, 10, 11],
#                 [12, 13, 16, 17, 18, 19, 14, 15]]
#     mesh.num_dof = 20
#     mesh.TYPE = [3, 3, 3, 3]
#     mesh.surf_of_ele = [0, 0, 0, 0]
#     # these next two arrays are incorret but are irrelevant for this test
#     mesh.bound_ele = np.array([[0, 0, 0], [0, 2, 2], [0, 3, 3], [1, 0, 0],
#                                [1, 1, 1], [1, 2, 2]])
#     mesh.nodes_in_bound_line = np.array([[0, 0, 1], [0, 1, 4], [1, 4, 5],
#                                          [2, 5, 2], [2, 2, 3], [3, 3, 0]])

#     fun1 = lambda x, y: x - .5
#     fun2 = lambda x, y: (-x + 2.5)
#     mat_zls0 = Material(E={-1: 2e11, 1: 1e11},
#                         nu={-1: .3, 1: .3},
#                         case='stress')
#     mat_zls1 = Material(E={-1: 5e11, 1: 1e11},
#                         nu={-1: .3, 1: .3},
#                         case='stress')
#     ZLS = [Create(fun1, [0, 4], [0, 1], material=mat_zls0, num_div=50),
#            Create(fun2, [0, 4], [0, 1], material=mat_zls1, num_div=50)]

#     MODEL = Build(mesh, zerolevelset=ZLS)

#     EPS0 = None
#     k = {}
#     element = {}
#     for eid, type in enumerate(MODEL.TYPE):
#         element[eid] = constructor(eid, MODEL, EPS0)
#         k[eid] = element[eid].stiffness_matrix()
#         print(element[eid].E)
#     assert element[0].E == [2e11, 1e11, 1e11, 2e11]
#     assert element[1].E == [1e11]*4
#     assert element[2].E == [1e11, 5e11, 5e11, 1e11]
#     assert element[3].E == [5e11]*4

