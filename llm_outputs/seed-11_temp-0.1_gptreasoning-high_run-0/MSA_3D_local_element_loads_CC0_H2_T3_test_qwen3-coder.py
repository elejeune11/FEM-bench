def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert_array_almost_equal(load_dofs_local, np.zeros(12))

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}
    L = 1.0
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    u_dofs_global = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    expected_axial_force = ele_info['E'] * ele_info['A'] / L
    assert abs(load_dofs_local[0] + expected_axial_force) < 1e-06
    assert abs(load_dofs_local[6] - expected_axial_force) < 1e-06
    u_dofs_global = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    expected_shear_force = 12 * ele_info['E'] * ele_info['I_z'] / L ** 3
    assert abs(load_dofs_local[1] + expected_shear_force) < 0.001
    assert abs(load_dofs_local[7] - expected_shear_force) < 0.001
    u_dofs_global = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    expected_torque = ele_info['G'] * ele_info['J'] / L if 'G' in ele_info else ele_info['E'] / (2 * (1 + ele_info['nu'])) * ele_info['J'] / L
    assert abs(load_dofs_local[3] + expected_torque) < 1e-06
    assert abs(load_dofs_local[9] - expected_torque) < 1e-06

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    ua = np.random.rand(12)
    ub = np.random.rand(12)
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert_array_almost_equal(fab, fa + fb)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 1.0, 0.0)
    u_dofs_global = np.random.rand(12)
    load_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    angle = np.pi / 4
    R = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    p1 = np.array([xi, yi, zi])
    p2 = np.array([xj, yj, zj])
    p1_rot = R @ p1
    p2_rot = R @ p2
    u_rotated = np.zeros(12)
    for i in range(2):
        u_idx = i * 6
        trans = u_dofs_global[u_idx:u_idx + 3]
        rot = u_dofs_global[u_idx + 3:u_idx + 6]
        u_rotated[u_idx:u_idx + 3] = R @ trans
        u_rotated[u_idx + 3:u_idx + 6] = R @ rot
    if 'local_z' in ele_info:
        ele_info_rot = ele_info.copy()
        ele_info_rot['local_z'] = R @ np.array(ele_info['local_z'])
    else:
        ele_info_rot = ele_info
    load_rotated = fcn(ele_info_rot, p1_rot[0], p1_rot[1], p1_rot[2], p2_rot[0], p2_rot[1], p2_rot[2], u_rotated)
    assert_array_almost_equal(load_original, load_rotated)