def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    rigid_translation = np.array([0.1, 0.05, -0.02, 0.0, 0.0, 0.0, 0.1, 0.05, -0.02, 0.0, 0.0, 0.0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, rigid_translation)
    assert_allclose(load_dofs_local, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    L = 1.0
    u_axial = np.array([0.001, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    expected_axial_force = ele_info['E'] * ele_info['A'] / L * 0.001
    assert_allclose(load_axial[0], expected_axial_force, rtol=1e-06)
    assert_allclose(load_axial[6], -expected_axial_force, rtol=1e-06)
    u_shear = np.array([0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0, -0.001, 0.0, 0.0, 0.0, 0.0])
    load_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    assert load_shear[1] != 0.0 or load_shear[7] != 0.0
    u_torsion = np.array([0.0, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    expected_torque = ele_info['E'] / (2 * (1 + ele_info['nu'])) * ele_info['J'] / L * 0.01
    assert_allclose(load_torsion[3], expected_torque, rtol=1e-05)
    assert_allclose(load_torsion[9], -expected_torque, rtol=1e-05)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_a = np.array([0.001, 0.0005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u_b = np.array([0.0, 0.0, 0.0002, 0.0, 0.005, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_a)
    load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_b)
    load_sum = load_a + load_b
    u_combined = u_a + u_b
    load_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    assert_allclose(load_combined, load_sum, rtol=1e-10, atol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global = np.array([0.001, 0.0005, 0.0002, 0.001, 0.002, 0.0005, 0.0005, 0.0003, 0.0001, 0.0005, 0.001, 0.0003])
    load_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    theta = np.pi / 6
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R = np.array([[cos_t, -sin_t, 0.0], [sin_t, cos_t, 0.0], [0.0, 0.0, 1.0]])
    p_i = np.array([xi, yi, zi])
    p_j = np.array([xj, yj, zj])
    p_i_rot = R @ p_i
    p_j_rot = R @ p_j
    (xi_rot, yi_rot, zi_rot) = p_i_rot
    (xj_rot, yj_rot, zj_rot) = p_j_rot
    u_dofs_rot = np.zeros(12)
    for k in range(2):
        u_dofs_rot[6 * k:6 * k + 3] = R @ u_dofs_global[6 * k:6 * k + 3]
        u_dofs_rot[6 * k + 3:6 * k + 6] = R @ u_dofs_global[6 * k + 3:6 * k + 6]
    load_rotated = fcn(ele_info, xi_rot, yi_rot, zi_rot, xj_rot, yj_rot, zj_rot, u_dofs_rot)
    assert_allclose(load_rotated, load_original, rtol=1e-09, atol=1e-12)