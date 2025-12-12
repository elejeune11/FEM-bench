def test_rigid_body_motion_zero_loads(fcn):
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global = [1.0, 2.0, 3.0, 0.1, 0.2, 0.3, 1.0, 2.0, 3.0, 0.1, 0.2, 0.3]
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert all((abs(f) < 1e-10 for f in load_dofs_local))

def test_unit_responses_axial_shear_torsion(fcn):
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_axial = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_axial)
    assert abs(load_axial[0] - ele_info['E'] * ele_info['A'] / 1.0) < 1e-06
    assert all((abs(f) < 1e-10 for f in load_axial[1:]))
    u_dofs_shear = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    load_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_shear)
    k_shear = 12 * ele_info['E'] * ele_info['I_z'] / 1.0 ** 3
    assert abs(load_shear[1] + k_shear) < 1e-06
    assert abs(load_shear[7] - k_shear) < 1e-06
    assert abs(load_shear[4] + k_shear * 1.0 / 2) < 1e-06
    assert abs(load_shear[10] - k_shear * 1.0 / 2) < 1e-06
    assert all((abs(f) < 1e-10 for (i, f) in enumerate(load_shear) if i not in [1, 4, 7, 10]))
    u_dofs_torsion = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_torsion)
    k_torsion = ele_info['E'] * ele_info['J'] / (1.0 * (1.0 + ele_info['nu']))
    assert abs(load_torsion[3] - k_torsion) < 1e-06
    assert abs(load_torsion[9] + k_torsion) < 1e-06
    assert all((abs(f) < 1e-10 for (i, f) in enumerate(load_torsion) if i not in [3, 9]))

def test_superposition_linearity(fcn):
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    u2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    u_sum = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    f1 = fcn(ele_info, xi, yi, zi, xj, yj, zj, u1)
    f2 = fcn(ele_info, xi, yi, zi, xj, yj, zj, u2)
    f_sum = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_sum)
    f_expected = [f1[i] + f2[i] for i in range(12)]
    for i in range(12):
        assert abs(f_sum[i] - f_expected[i]) < 1e-08

def test_coordinate_invariance_global_rotation(fcn):
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global = [0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.4, 0.5, 0.6, 0.04, 0.05, 0.06]
    local_z = [0.0, 0.0, 1.0]
    load_orig = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    (xi_rot, yi_rot, zi_rot) = R @ np.array([xi, yi, zi])
    (xj_rot, yj_rot, zj_rot) = R @ np.array([xj, yj, zj])
    local_z_rot = R @ np.array(local_z)
    u_rot = np.zeros(12)
    u_rot[0:3] = R @ np.array(u_dofs_global[0:3])
    u_rot[3:6] = R @ np.array(u_dofs_global[3:6])
    u_rot[6:9] = R @ np.array(u_dofs_global[6:9])
    u_rot[9:12] = R @ np.array(u_dofs_global[9:12])
    load_rot = fcn(ele_info, xi_rot, yi_rot, zi_rot, xj_rot, yj_rot, zj_rot, u_rot)
    for i in range(12):
        assert abs(load_orig[i] - load_rot[i]) < 1e-10