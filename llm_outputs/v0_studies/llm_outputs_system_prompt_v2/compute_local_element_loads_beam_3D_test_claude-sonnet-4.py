def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    translation = np.array([0.1, 0.2, 0.3])
    u_dofs_global = np.array([translation[0], translation[1], translation[2], 0, 0, 0, translation[0], translation[1], translation[2], 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-12)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-05
    I_z = 1e-05
    J = 2e-05
    L = 1.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    u_dofs_axial = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_axial)
    expected_axial_force = E * A / L
    assert np.isclose(load_axial[0], -expected_axial_force, rtol=1e-10)
    assert np.isclose(load_axial[6], expected_axial_force, rtol=1e-10)
    u_dofs_shear = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    load_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_shear)
    expected_shear_force = 12 * E * I_z / L ** 3
    assert np.isclose(load_shear[1], -expected_shear_force, rtol=1e-10)
    assert np.isclose(load_shear[7], expected_shear_force, rtol=1e-10)
    u_dofs_torsion = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_torsion)
    G = E / (2 * (1 + nu))
    expected_torsion_moment = G * J / L
    assert np.isclose(load_torsion[3], -expected_torsion_moment, rtol=1e-10)
    assert np.isclose(load_torsion[9], expected_torsion_moment, rtol=1e-10)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    ua = np.array([0.1, 0.05, 0.02, 0.01, 0.03, 0.02, 0.15, 0.08, 0.04, 0.02, 0.01, 0.03])
    ub = np.array([0.02, 0.03, 0.01, 0.005, 0.01, 0.015, 0.03, 0.04, 0.02, 0.01, 0.005, 0.02])
    uab = ua + ub
    load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    load_ab = fcn(ele_info, xi, yi, zi, xj, yj, zj, uab)
    assert np.allclose(load_ab, load_a + load_b, rtol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global = np.array([0.1, 0.05, 0.02, 0.01, 0.03, 0.02, 0.15, 0.08, 0.04, 0.02, 0.01, 0.03])
    load_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    pos_i = np.array([xi, yi, zi])
    pos_j = np.array([xj, yj, zj])
    pos_i_rot = R @ pos_i
    pos_j_rot = R @ pos_j
    local_z_rot = R @ ele_info['local_z']
    u_dofs_rot = np.zeros(12)
    for i in range(2):
        u_trans = u_dofs_global[6 * i:6 * i + 3]
        u_trans_rot = R @ u_trans
        u_dofs_rot[6 * i:6 * i + 3] = u_trans_rot
        u_rot = u_dofs_global[6 * i + 3:6 * i + 6]
        u_rot_rot = R @ u_rot
        u_dofs_rot[6 * i + 3:6 * i + 6] = u_rot_rot
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = local_z_rot
    load_rotated = fcn(ele_info_rot, pos_i_rot[0], pos_i_rot[1], pos_i_rot[2], pos_j_rot[0], pos_j_rot[1], pos_j_rot[2], u_dofs_rot)
    assert np.allclose(load_rotated, load_original, rtol=1e-10)