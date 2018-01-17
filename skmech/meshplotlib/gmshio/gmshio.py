"""writes the gmsh file"""
import numpy as np
import os


def write_field(field_value, mesh_file, field_name, ndim,
                time_value, time_step, exe_time,
                datatype='Node'):
    """Write vector field output file

    Parameters
    ----------
    field_value : dict
    mesh_file : string
    field_name : string
    ndim : int
    time_value : float
    time_step : int
    exe_time : float
    datatype : {'Node', 'Element'}

    """
    # Create file if it does not exist or if it was created during
    # another execution
    if os.path.isfile(f'{mesh_file}_out.msh'):
        # True if file already exist
        if exe_time > os.path.getmtime(f'{mesh_file}_out.msh'):
            # file was created before this execution, then rewrite
            with open(f'{mesh_file}_out.msh', 'w') as out, \
                 open(f'{mesh_file}.msh', 'r') as msh:
                # copy msh into out file
                out.writelines(l for l in msh)
        else:
            # file was created during this execution, do nothing
            pass
    else:
        # file does not exist, create
        with open(f'{mesh_file}_out.msh', 'w') as out, \
             open(f'{mesh_file}.msh', 'r') as msh:
            # copy msh into out file
            out.writelines(l for l in msh)

    header = f"""
${datatype}Data
1
"{field_name}"
1
{time_value}
3
{time_step}
{ndim}
{len(field_value)}
"""
    with open(f'{mesh_file}_out.msh', 'a') as out:

        out.write(header)
        for nid, value in field_value.items():

            try:
                if len(value) > 1:
                    value_str = ''
                    for i in range(ndim):
                        value_str += str(value[i]) + ' '
            except TypeError:
                # value is a scalar and does not have len()
                value_str = str(value)

            out.write(f'{nid} {value_str}\n')

        out.write(f'$End{datatype}Data')


if __name__ == '__main__':
    with open('test.msh', 'w') as test_file:
        test_mesh = """$MeshFormat
2.2 0 8
$EndMeshFormat
$Nodes
29
1 0.05 0 0
2 0.18 0 0
3 0.18 0.1 0
4 0 0.1 0
5 0 0.05 0
6 0.08249999999993463 0 0
7 0.1149999999998537 0 0
8 0.147499999999919 0 0
9 0.18 0.02499999999994222 0
10 0.18 0.04999999999986842 0
11 0.18 0.07499999999995359 0
12 0.1350000000000678 0.1 0
13 0.09000000000024656 0.1 0
14 0.04500000000015603 0.1 0
15 0 0.08750000000002109 0
16 0 0.07500000000006782 0
17 0 0.06250000000004673 0
18 0.01913417161821371 0.04619397662558123 0
19 0.03535533905925894 0.03535533905939582 0
20 0.04619397662554421 0.01913417161830309 0
21 0.04500000000012328 0.08750000000003391 0
22 0.1350000000001233 0.07499999999993422 0
23 0.06267766952975275 0.06767766952969792 0
24 0.03206708580916849 0.06684698831281399 0
25 0.1025000000000501 0.05 0
26 0.07434698831280107 0.03456708580915155 0
27 0.1412500000000211 0.03749999999997111 0
28 0.03000000000008764 0.09166666666668713 0
29 0.150000000000073 0.0833333333332927 0
$EndNodes
$Elements
30
1 1 2 1 1 1 6
2 1 2 1 1 6 7
3 1 2 1 1 7 8
4 1 2 1 1 8 2
5 1 2 4 2 2 9
6 1 2 4 2 9 10
7 1 2 4 2 10 11
8 1 2 4 2 11 3
9 1 2 2 4 4 15
10 1 2 2 4 15 16
11 1 2 2 4 16 17
12 1 2 2 4 17 5
13 3 2 3 1 5 18 24 17
14 3 2 3 1 18 19 23 24
15 3 2 3 1 24 23 13 21
16 3 2 3 1 17 24 21 16
17 3 2 3 1 1 6 26 20
18 3 2 3 1 6 7 25 26
19 3 2 3 1 26 25 13 23
20 3 2 3 1 20 26 23 19
21 3 2 3 1 13 25 27 22
22 3 2 3 1 25 7 8 27
23 3 2 3 1 27 8 2 9
24 3 2 3 1 22 27 9 10
25 3 2 3 1 16 21 28 15
26 3 2 3 1 21 13 14 28
27 3 2 3 1 15 28 14 4
28 3 2 3 1 10 11 29 22
29 3 2 3 1 11 3 12 29
30 3 2 3 1 22 29 12 13
$EndElements
"""
        test_file.write(test_mesh)
    u = {1: np.array([0.0007109, 0.]),
         2: np.array([0.001, 0.]),
         3: np.array([0.001, -0.00013852]),
         4: np.array([0., -0.00046987]),
         5: np.array([0., -0.00035611]),
         6: np.array([0.00076201, 0.]),
         7: np.array([0.00083113, 0.]),
         8: np.array([0.00090883, 0.]),
         9: np.array([1.00000000e-03, -1.80711111e-05]),
         10: np.array([1.00000000e-03, -4.66897349e-05]),
         11: np.array([1.00000000e-03, -8.95448654e-05]),
         12: np.array([0.00077269, -0.00014547]),
         13: np.array([0.00050693, -0.00015763]),
         14: np.array([0.00020967, -0.0003355]),
         15: np.array([0., -0.00044474]),
         16: np.array([0., -0.00042611]),
         17: np.array([0., -0.0004041]),
         18: np.array([0.00028406, -0.00026887]),
         19: np.array([0.00048873, -0.00017441]),
         20: np.array([6.49925139e-04, -8.37454412e-05]),
         21: np.array([0.00025606, -0.00029968]),
         22: np.array([7.96275022e-04, -8.72806150e-05]),
         23: np.array([0.0004358, -0.00015991]),
         24: np.array([0.00023824, -0.0002902]),
         25: np.array([7.01365215e-04, -5.06101326e-05]),
         26: np.array([6.54214389e-04, -3.53889208e-05]),
         27: np.array([8.63397141e-04, -3.05876099e-05]),
         28: np.array([0.00015529, -0.00036516]),
         29: np.array([0.00085572, -0.00010478])}
    u2 = {nid: [ux * 2, uy * 1.2] for nid, [ux, uy] in u.items()}
    u3 = {nid: [ux * 2, uy * 1.2] for nid, [ux, uy] in u2.items()}

    sig = {5: np.array([3.95494165e+08, 1.30459435e+08, 9.63761480e+07]),
           18: np.array([86958751.3855, 13830365.96491699, -40328139.203]),
           24: np.array([53049649.1548, -7787495.97851688, -5005342.732]),
           17: np.array([7781510.8591, -10046565.94870567, 26060960.570]),
           19: np.array([65932828.6886, 8981008.37896344, -7605102.820]),
           23: np.array([57364248.4389, 993899.3101743, -272804.161]),
           13: np.array([45518656.9422, 1203977.36989567, -1155715.550]),
           21: np.array([60984696.4123, 17905502.21089627, 9254379.616]),
           16: np.array([32943141.8836, 1893225.90848076, 9849927.853]),
           1: np.array([-32451770.0174, -99553402.345294, -25924835.3161]),
           6: np.array([13358669.48973, 9012221.27834559, -9029027.9805]),
           26: np.array([15899143.6825, -5907993.66500521, -10453092.924]),
           20: np.array([-988673.8246, -51194840.43222095, 1427448.835]),
           7: np.array([10705993.08271, -4085222.40021804, -7080464.5720]),
           25: np.array([15095171.2235, -4981925.17162373, -8242168.766]),
           27: np.array([28015915.9624, 4035649.93063616, -1581842.525]),
           22: np.array([52305564.4543, 10193215.97514246, -1044130.716]),
           8: np.array([9387876.534755, -1868698.45009, -2125887.7012706]),
           2: np.array([13279344.89350, 1281990.94761415, 2843190.4765]),
           9: np.array([32236711.79345, 9226753.12382996, 3362354.6482]),
           10: np.array([46713934.5611, 10184645.55052388, 2390821.314]),
           28: np.array([20253598.4228, -60571405.56588, -59915889.017]),
           15: np.array([40730635.9660, 27596202.30219314, 34552838.767]),
           14: np.array([36574827.0814, -12908863.73360975, 5748126.210]),
           4: np.array([38616791.34947, 1836699.01406159, 5459636.3970]),
           11: np.array([50004186.7469, 17690882.86809701, 8097989.234]),
           29: np.array([-26948622.2322, -50229651.886583, -20533988.588]),
           3: np.array([37854307.03757, -166570.33098567, -427146.3798]),
           12: np.array([70137277.9780, 18367441.25470823, 7868738.264])}

    s11 = {nid: sxx for nid, [sxx, _, _] in sig.items()}
    s112 = {nid: sxx * 1.2 for nid, [sxx, _, _] in sig.items()}
    s113 = {nid: sxx * 1.6 for nid, [sxx, _, _] in sig.items()}

    import time
    exe_time = time.time()
    write_field(u, 'test', 'Displacement', 2, 0, 0, exe_time)
    write_field(u2, 'test', 'Displacement', 2, 0.2, 1, exe_time)
    write_field(u3, 'test', 'Displacement', 2, 0.4, 2, exe_time)
    write_field(s11, 'test', 's11', 1, 0, 0, exe_time)
    write_field(s112, 'test', 's11', 1, 0.2, 1, exe_time)
    write_field(s113, 'test', 's11', 1, 0.4, 2, exe_time)
