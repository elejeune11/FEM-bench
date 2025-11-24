def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    translation = np.array([0.5, -0.3, 0.7])
    u_dofs_global = np.array([translation[0], translation[1], translation[2], 0, 0, 0, translation[0], translation[1], translation[2], 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    L = 2.0
    u_dofs_axial = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_axial)
    expected_axial_force = ele_info['E'] * ele_info['A'] / L
    assert np.abs(load_axial[0] + expected_axial_force) < 1e-06
    assert np.abs(load_axial[6] - expected_axial_force) < 1e-06
    u_dofs_shear = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    load_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_shear)
    shear_stiffness = 12 * ele_info['E'] * ele_info['I_z'] / L ** 3
    expected_shear = shear_stiffness
    assert np.abs(load_shear[1] + expected_shear) < 1e-06
    assert np.abs(load_shear[7] - expected_shear) < 1e-06
    u_dofs_torsion = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_torsion)
    G = ele_info['E'] / (2 * (1 + ele_info['nu']))
    torsional_stiffness = G * ele_info['J'] / L
    expected_torque = torsional_stiffness
    assert np.abs(load_torsion[3] + expected_torque) < 1e-06
    assert np.abs(load_torsion[9] - expected_torque) < 1e-06

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 210000000000.0, 'nu': 0.28, 'A': 0.015, 'I_y': 2e-05, 'I_z': 3e-05, 'J': 4e-05, 'local_z': np.array([0, 1, 0])}
    (xi, yi, zi) = (1.0, 2.0, 3.0)
    (xj, yj, zj) = (4.0, 5.0, 6.0)
    np.random.seed(42)
    u_a = np.random.randn(12) * 0.01
    u_b = np.random.randn(12) * 0.01
    u_combined = u_a + u_b
    load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_a)
    load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_b)
    load_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    assert np.allclose(load_combined, load_a + load_b, rtol=1e-10, atol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    np.random.seed(123)
    u_dofs_global = np.random.randn(12) * 0.001
    load_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    theta = np.pi / 4
    phi = np.pi / 6
    Rz = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    R = Rz @ Ry
    pi_rot = R @ np.array([xi, yi, zi])
    pj_rot = R @ np.array([xj, yj, zj])
    u_dofs_rotated = np.zeros(12)
    u_dofs_rotated[0:3] = R @ u_dofs_global[0:3]
    u_dofs_rotated[3:6] = R @ u_dofs_global[3:6]
    u_dofs_rotated[6:9] = R @ u_dofs_global[6:9]
    u_dofs_rotated[9:12] = R @ u_dofs_global[9:12]
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = R @ ele_info['local_z']
    load_rotated = fcn(ele_info_rot, pi_rot[0], pi_rot[1], pi_rot[2], pj_rot[0], pj_rot[1], pj_rot[2], u_dofs_rotated)
    assert np.allclose(load_rotated, load_original, rtol=1e-10, atol=1e-12)