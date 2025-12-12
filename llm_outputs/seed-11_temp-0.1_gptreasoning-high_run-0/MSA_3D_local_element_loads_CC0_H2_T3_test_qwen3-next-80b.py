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
    k_torsion = ele_info['E'] * ele_info['J'] / 1.0
    assert abs(load_torsion[3] + k_torsion) < 1e-06
    assert abs(load_torsion[9] - k_torsion) < 1e-06
    assert all((abs(f) < 1e-10 for (i, f) in enumerate(load_torsion) if i not in [3, 9]))

def test_superposition_linearity(fcn):
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u1 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    u2 = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    u_sum = [u1[i] + u2[i] for i in range(12)]
    f1 = fcn(ele_info, xi, yi, zi, xj, yj, zj, u1)
    f2 = fcn(ele_info, xi, yi, zi, xj, yj, zj, u2)
    f_sum = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_sum)
    for i in range(12):
        assert abs(f_sum[i] - (f1[i] + f2[i])) < 1e-06

def test_coordinate_invariance_global_rotation(fcn):
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global = [0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.15, 0.25, 0.35, 0.015, 0.025, 0.035]
    local_z = [0.0, 0.0, 1.0]
    load_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    (xi_rot, yi_rot, zi_rot) = R @ np.array([xi, yi, zi])
    (xj_rot, yj_rot, zj_rot) = R @ np.array([xj, yj, zj])
    local_z_rot = R @ np.array(local_z)
    u_trans = np.array(u_dofs_global[0:6:1])
    u_rot = np.array(u_dofs_global[6:12:1])
    u_trans_rot = np.zeros(6)
    u_rot_rot = np.zeros(6)
    for i in range(3):
        u_trans_rot[i] = R[i, 0] * u_trans[0] + R[i, 1] * u_trans[1] + R[i, 2] * u_trans[2]
        u_trans_rot[i + 3] = R[i, 0] * u_trans[3] + R[i, 1] * u_trans[4] + R[i, 2] * u_trans[5]
        u_rot_rot[i] = R[i, 0] * u_rot[0] + R[i, 1] * u_rot[1] + R[i, 2] * u_rot[2]
        u_rot_rot[i + 3] = R[i, 0] * u_rot[3] + R[i, 1] * u_rot[4] + R[i, 2] * u_rot[5]
    u_dofs_rot = np.concatenate([u_trans_rot, u_rot_rot])
    load_rotated = fcn(ele_info, xi_rot, yi_rot, zi_rot, xj_rot, yj_rot, zj_rot, u_dofs_rot)
    for i in range(12):
        assert abs(load_original[i] - load_rotated[i]) < 1e-10