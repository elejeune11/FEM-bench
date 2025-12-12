def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    translation = np.array([0.5, 0.3, 0.2])
    u_dofs_global = np.array([translation[0], translation[1], translation[2], 0.0, 0.0, 0.0, translation[0], translation[1], translation[2], 0.0, 0.0, 0.0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-10), 'Rigid body translation should produce zero internal forces'

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    L = 2.0
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 0.0001
    I_z = 0.0001
    J = 0.0002
    G = E / (2 * (1 + nu))
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    u_axial = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    expected_axial_force = E * A / L
    assert np.isclose(load_axial[0], -expected_axial_force, rtol=1e-06), 'Axial force at node 1 should be -EA/L for unit extension'
    assert np.isclose(load_axial[6], expected_axial_force, rtol=1e-06), 'Axial force at node 2 should be EA/L for unit extension'
    u_shear = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0])
    load_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    expected_shear = 12 * E * I_z / L ** 3
    assert np.isclose(load_shear[1], -expected_shear, rtol=1e-06), 'Shear force at node 1 should be negative for positive v2'
    assert np.isclose(load_shear[7], expected_shear, rtol=1e-06), 'Shear force at node 2 should be positive for positive v2'
    u_torsion = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    expected_torsion = G * J / L
    assert np.isclose(load_torsion[3], -expected_torsion, rtol=1e-06), 'Torsional moment at node 1 should be -GJ/L for unit twist'
    assert np.isclose(load_torsion[9], expected_torsion, rtol=1e-06), 'Torsional moment at node 2 should be GJ/L for unit twist'

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (3.0, 0.0, 0.0)
    ua = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
    ub = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.002, 0.0, 0.0, 0.0, 0.01])
    u_combined = ua + ub
    load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    load_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    assert np.allclose(load_combined, load_a + load_b, rtol=1e-10), 'Superposition should hold: f(ua + ub) = f(ua) + f(ub)'
    alpha = 2.5
    load_scaled = fcn(ele_info, xi, yi, zi, xj, yj, zj, alpha * ua)
    assert np.allclose(load_scaled, alpha * load_a, rtol=1e-10), 'Homogeneity should hold: f(alpha * u) = alpha * f(u)'

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info_original = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    u_dofs_original = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.002, 0.0, 0.0, 0.0, 0.01])
    load_original = fcn(ele_info_original, xi, yi, zi, xj, yj, zj, u_dofs_original)
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    node_i = R @ np.array([xi, yi, zi])
    node_j = R @ np.array([xj, yj, zj])
    (xi_rot, yi_rot, zi_rot) = node_i
    (xj_rot, yj_rot, zj_rot) = node_j
    local_z_rot = R @ ele_info_original['local_z']
    ele_info_rotated = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': local_z_rot}
    u1_trans = R @ u_dofs_original[0:3]
    u1_rot = R @ u_dofs_original[3:6]
    u2_trans = R @ u_dofs_original[6:9]
    u2_rot = R @ u_dofs_original[9:12]
    u_dofs_rotated = np.concatenate([u1_trans, u1_rot, u2_trans, u2_rot])
    load_rotated = fcn(ele_info_rotated, xi_rot, yi_rot, zi_rot, xj_rot, yj_rot, zj_rot, u_dofs_rotated)
    assert np.allclose(load_original, load_rotated, rtol=1e-06), 'Local internal loads should be invariant under global rotation of the entire system'