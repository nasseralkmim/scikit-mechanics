"""Test quad enr element

Tests:

1. Element material properties matrix and reinforcement
  1.1 Test element material for non enriched elements in xfem analysis
2. Stiffness matrix
3. Dof for when there is more than one zero level set

"""
import numpy as np
import skmech


def test_element_stiffness():
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
                10: [15, 2, 12, 1, 1],
                20: [15, 2, 13, 4, 4],
                13: [1, 2, 3, 30, 6, 7],
                30: [1, 2, 7, 2, 3, 7],
                4: [1, 2, 7, 2, 7, 4],
                1: [3, 2, 11, 10, 1, 2, 3, 4],
                2: [3, 2, 11, 10, 2, 5, 8, 3],
                3: [3, 2, 11, 10, 5, 6, 7, 8],
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
        k[eid] = ele.stiffness_matrix()[ele.id_m]  # extract non augmentbed

    assert emat[3] == [109890109890.1099,
                       109890109890.1099,
                       109890109890.1099,
                       109890109890.1099]
    assert emat[2] == [219780219780.2198,
                       109890109890.1099,
                       109890109890.1099,
                       219780219780.2198]
    k_std_ele1 = np.array(
        [[1.154, 0.481, -0.769, 0.096, -0.577, -0.481, 0.192, -0.096],
         [0.481, 1.154, -0.096, 0.192, -0.481, -0.577, 0.096, -0.769],
         [-0.769, -0.096, 1.154, -0.481, 0.192, 0.096, -0.577, 0.481],
         [0.096, 0.192, -0.481, 1.154, -0.096, -0.769, 0.481, -0.577],
         [-0.577, -0.481, 0.192, -0.096, 1.154, 0.481, -0.769, 0.096],
         [-0.481, -0.577, 0.096, -0.769, 0.481, 1.154, -0.096, 0.192],
         [0.192, 0.096, -0.577, 0.481, -0.769, -0.096, 1.154, -0.481],
         [-0.096, -0.769, 0.481, -0.577, 0.096, 0.192, -0.481, 1.154]])
    assert np.allclose(k[1][:8, :8] / 1e9, k_std_ele1, atol=1e-2)
    k_enr_ele1 = np.array(
        [[129.915, 0., 49.573, -0.], [-0., 70.085, -0., -18.803],
         [49.573, -0., 129.915, 0.], [0., -18.803, 0., 70.085]])
    assert np.allclose(k[1][8:, 8:] / 1e9 * 3600, k_enr_ele1, atol=1e-2)


def test_dof():
    """test dof for 2 zero level set thtat don't share an element with 4 ele"""
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
    """test dof for two zero level sets in 2 elements"""
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

    # element dof numbering following CCW for standards and following
    # the sorted enr node order for each zero level set
    assert dof[1] == [1, 2, 3, 4, 5, 6, 7, 8,
                      13, 14, 15, 16, 17, 18, 19, 20,  # first zls
                      21, 22, 23, 24]                  # second
    assert dof[2] == [3, 4, 9, 10, 11, 12, 5, 6,
                      15, 16, 17, 18,  # node 2, 3 enr dof for first zls
                      21, 22, 23, 24, 25, 26, 27, 28]  # nodes 2,3,5,6 2nd zls


def test_dof3():
    """test dof for 4 zero level sets in separate 4 elements
    also tests for matrix material properties.

    """
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
    # -1 is reinforcement, 1 is matrix
    mat = skmech.Material(E={-1: 2e11, 1: 1e11}, nu={-1: 0.3, 1: 0.3},
                          case='stress')
    model = skmech.Model(msh, zerolevelset=zls, material=mat,
                         thickness=0.01)

    E = {}
    dof = {}
    for eid, [etype, *_] in model.elements.items():
        ele = skmech.constructor(eid, etype, model)
        dof[eid] = ele.dof
        E[eid] = ele.E

    assert dof[1] == [1, 2, 3, 4, 5, 6, 7, 8, 19, 20, 21, 22, 23, 24, 25,
                      26, 27, 28, 29, 30, 35, 36, 43, 44, 45, 46]
    assert dof[2] == [3, 4, 9, 10, 11, 12, 5, 6, 21, 22, 23, 24, 27, 28, 29,
                      30, 31, 32, 33, 34, 35, 36, 37, 38, 43, 44]
    assert dof[3] == [5, 6, 11, 12, 13, 14, 15, 16, 23, 24, 29, 30, 33, 34, 35,
                      36, 37, 38, 39, 40, 41, 42, 43, 44, 47, 48]
    assert dof[4] == [7, 8, 5, 6, 15, 16, 17, 18, 23, 24, 25, 26, 29, 30, 35,
                      36, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50]
    assert E[1] == [2e11, 1e11, 1e11, 1e11]
    assert E[2] == [1e11, 2e11, 1e11, 1e11]
    assert E[3] == [1e11, 1e11, 2e11, 1e11]
    assert E[4] == [1e11, 1e11, 1e11, 2e11]


def test_material_nonenriched_element():
    """test if material is properly asigned for non enriched elements when in 
    xfem analysis with 3 elements

    """
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
                8: [3, 1, 0]
            }
            self.elements = {
                1: [3, 2, 11, 10, 1, 2, 3, 4],
                2: [3, 2, 11, 10, 2, 5, 6, 3],
                3: [3, 2, 11, 10, 5, 7, 8, 6]
            }
    msh = Mesh()
    func = lambda x, y: x - 0.5
    zls = skmech.xfem.ZeroLevelSet(func, [0, 3], [0, 1], num_div=50)
    mat = skmech.Material(E={-1: 2.22, 1: 1.11}, nu={-1: 0.3, 1: 0.3})
    model = skmech.Model(msh, zerolevelset=zls, material=mat, thickness=0.01)

    E = {}
    for eid, [etype, *_] in model.elements.items():
        ele = skmech.constructor(eid, etype, model)
        E[eid] = ele.E
    assert E[3] == 1.11
    assert E[1] == [2.22, 1.11, 1.11, 2.22]
    assert E[2] == [1.11, 1.11, 1.11, 1.11]
