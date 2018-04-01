import numpy as np
import skmech
# from elastopy.xfem.zerolevelset import Create
# from elastopy.model import model


# def test_1zerolevelset():
#     class Mesh():
#         pass
#     mesh = Mesh()
#     mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
#     mesh.CONN = np.array([[0, 1, 2, 3]])
#     mesh.num_ele = 1
#     mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7]]
#     mesh.num_dof = 8

#     def func(x, y):
#         return x - .5
#     zls = Create(func, [0, 1], [0, 1], num_div=3)

#     model = Build(mesh, zerolevelset=[zls])
#     assert (model.zerolevelset[0].phi == [-.5, .5, .5, -.5]).all()
#     assert model.zerolevelset[0].enriched_nodes == [0, 1, 2, 3]
#     assert model.num_enr_dof == 8


# def test_2zerolevelset():
#     class Mesh():
#         pass
#     mesh = Mesh()
#     mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
#                          [2, 0], [2, 1],
#                          [2, 2], [1, 2],
#                          [0, 2]])
#     mesh.CONN = np.array([[0, 1, 2, 3], [1, 4, 5, 2], [2, 5, 6, 7],
#                           [3, 2, 7, 8]])
#     mesh.num_ele = 4
#     mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7],
#                 [2, 3, 8, 9, 4, 5, 10, 11],
#                 [4, 5, 10, 11, 12, 13, 14, 15],
#                 [6, 7, 4, 5, 14, 15, 16, 17]]
#     mesh.num_dof = 18

#     def func(x, y):
#         return (x-1)**2 + (y-1)**2 - .5**2
#     zls = Create(func, [0, 2], [0, 2], num_div=10)
#     model = Build(mesh, zerolevelset=[zls])

#     assert (model.zerolevelset[0].enriched_nodes ==
#             [0, 1, 2, 3, 4, 5, 6, 7, 8])
#     print(model.zerolevelset[0].enriched_elements)
#     assert model.zerolevelset[0].enriched_elements == [0, 1, 2, 3]
#     assert model.enriched_elements == [0, 1, 2, 3]


# def test_3zerolevelset():
#     class Mesh():
#         pass
#     mesh = Mesh()
#     mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
#                          [2, 0], [2, 1]])
#     mesh.CONN = np.array([[0, 1, 2, 3], [1, 4, 5, 2]])
#     mesh.num_ele = 2
#     mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7],
#                 [2, 3, 8, 9, 10, 11, 4, 5]]
#     mesh.num_dof = 12

#     fun1 = lambda x, y: x - .5
#     fun2 = lambda x, y: (-x + 1.5)
#     zls = [Create(fun1, [0, 2], [0, 1], num_div=10),
#            Create(fun2, [0, 2], [0, 1], num_div=10)]

#     model = Build(mesh, zerolevelset=zls)
#     assert set(model.zerolevelset[0].enriched_nodes) == set([0, 1, 2, 3])
#     assert set(model.zerolevelset[1].enriched_nodes) == set([1, 2, 4, 5])
#     assert model.zerolevelset[0].discontinuity_elements == [0]
#     assert model.zerolevelset[1].discontinuity_elements == [1]

#     # includes dof for each element and for each zero level set
#     # the order is defined by the zero level set order
#     # and by the enriched nodes order
#     assert model.DOF == [[0, 1, 2, 3, 4, 5, 6, 7,
#                           12, 13, 14, 15, 16, 17, 18, 19,
#                           20, 21, 22, 23],
#                          [2, 3, 8, 9, 10, 11, 4, 5,
#                           14, 15, 16, 17,
#                           20, 21, 22, 23, 24, 25, 26, 27]]
#     # numbering order defined by enriched nodes
#     assert model.zerolevelset[0].enriched_dof[0] == [12, 13, 14, 15, 16, 17,
#                                                      18, 19]
#     assert model.zerolevelset[0].enriched_dof[1] == [14, 15, 16, 17]
#     assert model.zerolevelset[1].enriched_dof[0] == [20, 21, 22, 23]
#     assert model.num_enr_dof == 2*8
#     assert model.enriched_elements == [0, 1]


# def test_4zerolevelset():
#     class Mesh():
#         pass
#     mesh = Mesh()
#     mesh.XYZ = np.array([[0, 0], [1, 0], [1, 1], [0, 1],
#                          [2, 0], [2, 1],
#                          [3, 0], [3, 1],
#                          [3, 2], [2, 2], [1, 2], [0, 2]])
#     mesh.CONN = np.array([[0, 1, 2, 3], [1, 4, 5, 2],
#                           [4, 6, 7, 5], [5, 7, 8, 9],
#                           [2, 5, 9, 10], [3, 2, 10, 11]])
#     mesh.num_ele = 6
#     mesh.DOF = [[0, 1, 2, 3, 4, 5, 6, 7],
#                 [2, 3, 8, 9, 10, 11, 4, 5],
#                 [8, 9, 12, 13, 14, 15, 10, 11],
#                 [10, 11, 14, 15, 16, 17, 18, 19],
#                 [4, 5, 10, 11, 18, 19, 20, 21],
#                 [9, 10, 4, 5, 20, 21, 22, 23]]

#     mesh.num_dof = 22

#     fun = lambda x, y: (x-3)**2 + (y-2)**2 - .3**2
#     zls = [Create(fun, [0, 3], [0, 2], num_div=15)]

#     model = Build(mesh, zerolevelset=zls)
#     assert model.zerolevelset[0].discontinuity_elements == [3]
#     assert model.zerolevelset[0].enriched_nodes == [5, 7, 8, 9]
#     assert model.zerolevelset[0].enriched_elements == [1, 2, 3, 4]
#     # dof numbering according to enriched nodes order - sorted
#     assert model.zerolevelset[0].enriched_dof[3] == [24, 25, 26, 27,
#                                                      28, 29, 30, 31]
#     assert model.enriched_elements == [1, 2, 3, 4]
#     assert model.enriched_nodes == [5, 7, 8, 9]  # sorted


# def test_5zerolevelset():
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

#     fun1 = lambda x, y: x - .5
#     fun2 = lambda x, y: (-x + 3.5)
#     zls = [Create(fun1, [0, 4], [0, 1], num_div=50),
#            Create(fun2, [0, 4], [0, 1], num_div=50)]
#     model = Build(mesh, zerolevelset=zls)

#     assert model.zerolevelset[0].enriched_elements == [0, 1]
#     assert model.zerolevelset[1].enriched_elements == [2, 3]
#     assert model.zerolevelset[0].enriched_nodes == [0, 1, 2, 3]
#     assert model.zerolevelset[1].enriched_nodes == [6, 7, 8, 9]  # sorted

#     # enriched dof is numbering according to enriched_nodes order
#     assert model.DOF[0] == [0, 1, 2, 3, 4, 5, 6, 7,
#                             20, 21, 22, 23, 24, 25, 26, 27]
#     assert model.DOF[1] == [2, 3, 8, 9, 10, 11, 4, 5,
#                             22, 23, 24, 25]
#     assert model.DOF[3] == [12, 13, 16, 17, 18, 19, 14, 15,
#                             28, 29, 30, 31, 32, 33, 34, 35]
#     assert model.DOF[2] == [8, 9, 12, 13, 14, 15, 10, 11,
#                             28, 29, 30, 31]
#     assert list(model.zerolevelset[0].phi) == [-0.13265306122448967,
#                                                0.11734693877551011,
#                                                0.11734693877551011,
#                                                -0.13265306122448967,
#                                                0.36734693877550939,
#                                                0.36734693877550939,
#                                                0.6173469387755075,
#                                                0.6173469387755075,
#                                                0.86734693877552016,
#                                                0.86734693877552016]
#     assert list(model.zerolevelset[1].phi) == [0.86734693877552016,
#                                                0.6173469387755075,
#                                                0.6173469387755075,
#                                                0.86734693877552016,
#                                                0.36734693877550939,
#                                                0.36734693877550939,
#                                                0.11734693877551006,
#                                                0.11734693877551004,
#                                                -0.13265306122448967,
#                                                -0.13265306122448967]
def test_xyz():
        class Mesh():
            pass

        msh = Mesh()
        msh.nodes = {
            1: [0, 0, 0],
            2: [1, 0, 0],
            3: [1, 1, 0],
            4: [0, 1, 0],
            5: [.5, 0, 0],
            6: [1, .5, 0],
            7: [.5, 1, 0],
            8: [0, .5, 0],
            9: [.4, .6]
        }
        msh.elements = {
            1: [15, 2, 12, 1, 1],
            2: [15, 2, 13, 2, 2],
            3: [1, 2, 7, 2, 2, 6],
            4: [1, 2, 7, 2, 6, 3],
            7: [1, 2, 5, 4, 4, 8],
            8: [1, 2, 5, 4, 8, 1],
            9: [3, 2, 11, 10, 1, 5, 9, 8],
            10: [3, 2, 11, 10, 5, 2, 6, 9],
            11: [3, 2, 11, 10, 9, 6, 3, 7],
            12: [3, 2, 11, 10, 8, 9, 7, 4]
        }
        model = skmech.Model(msh, displacement_bc={12: (0, 0),
                                                   13: (None, 0)})
        xyz = np.array(list(msh.nodes[n][:2] for n in msh.nodes.keys()))
        print(model.id_r, model.id_f)


test_xyz()
