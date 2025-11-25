def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 2e-05, 'J': 1e-05, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    u_dofs_global = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0.1, 0.2, 0.3, 0, 0, 0])
    result = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(result, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 2e-05, 'J': 1e-05, 'local_z': [0, 0, 1]}
    L = 2.0
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (L, 0, 0)
    u_axial = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    result_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    expected_axial_force = ele_info['E'] * ele_info['A'] / L
    assert abs(result_axial[6] - expected_axial_force) < 1e-10
    assert abs(result_axial[0] + expected_axial_force) < 1e-10
    u_shear = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    result_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    EI = ele_info['E'] * ele_info['I_z']
    expected_shear_force = 12 * EI / L ** 3
    assert abs(result_shear[7] - expected_shear_force) < 1e-10
    assert abs(result_shear[1] + expected_shear_force) < 1e-10
    u_torsion = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    result_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    G = ele_info['E'] / (2 * (1 + ele_info['nu']))
    expected_torsion_moment = G * ele_info['J'] / L
    assert abs(result_torsion[9] - expected_torsion_moment) < 1e-10
    assert abs(result_torsion[3] + expected_torsion_moment) < 1e-10

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 2e-05, 'J': 1e-05, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    ua = np.array([0.001, 0.002, 0.003, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.004, 0.005, 0.006])
    ub = np.array([0.0005, 0.0015, 0.0025, 0.0005, 0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0035, 0.0045, 0.0055])
    u_combined = ua + ub
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    f_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    assert np.allclose(f_combined, fa + fb, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 2e-05, 'J': 1e-05, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    u_dofs_global = np.array([0.001, 0.002, 0.003, 0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.004, 0.005, 0.006])
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    coords_i_rot = R @ np.array([xi, yi, zi])
    coords_j_rot = R @ np.array([xj, yj, zj])
    local_z_rot = R @ np.array(ele_info['local_z'])
    u_rot = np.zeros(12)
    for i in range(0, 12, 3):
        u_rot[i:i + 3] = R @ u_dofs_global[i:i + 3]
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = local_z_rot.tolist()
    result_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    result_rotated = fcn(ele_info_rot, *coords_i_rot, *coords_j_rot, u_rot)
    assert np.allclose(result_original, result_rotated, atol=1e-10)