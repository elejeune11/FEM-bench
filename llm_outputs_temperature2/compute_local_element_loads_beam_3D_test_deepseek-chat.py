def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    u_rigid = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0.1, 0.2, 0.3, 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_rigid)
    assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    L = 2.0
    E = 200000000000.0
    A = 0.01
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.67e-05
    G = E / (2 * (1 + 0.3))
    ele_info = {'E': E, 'nu': 0.3, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (L, 0, 0)
    u_axial = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    expected_axial_force = E * A / L
    assert np.isclose(load_axial[6], -expected_axial_force)
    assert np.isclose(load_axial[0], expected_axial_force)
    u_shear_y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    load_shear_y = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y)
    assert not np.allclose(load_shear_y, np.zeros(12))
    u_torsion = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    expected_torsion_moment = G * J / L
    assert np.isclose(load_torsion[9], -expected_torsion_moment)
    assert np.isclose(load_torsion[3], expected_torsion_moment)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (3, 0, 0)
    ua = np.array([0.001, 0.002, 0.003, 0.01, 0.02, 0.03, 0.004, 0.005, 0.006, 0.04, 0.05, 0.06])
    ub = np.array([0.0005, -0.001, 0.002, -0.005, 0.01, -0.015, 0.001, -0.002, 0.003, -0.02, 0.03, -0.04])
    u_combined = ua + ub
    load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    load_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    assert np.allclose(load_combined, load_a + load_b, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info_orig = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': [0, 0, 1]}
    (xi_orig, yi_orig, zi_orig) = (0, 0, 0)
    (xj_orig, yj_orig, zj_orig) = (2, 1, 0)
    u_dofs_global_orig = np.array([0.001, 0.002, 0.003, 0.01, 0.02, 0.03, 0.004, 0.005, 0.006, 0.04, 0.05, 0.06])
    load_orig = fcn(ele_info_orig, xi_orig, yi_orig, zi_orig, xj_orig, yj_orig, zj_orig, u_dofs_global_orig)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    coords_i_rot = R @ np.array([xi_orig, yi_orig, zi_orig])
    coords_j_rot = R @ np.array([xj_orig, yj_orig, zj_orig])
    local_z_rot = R @ np.array([0, 0, 1])
    u_rotated = np.zeros(12)
    u_rotated[0:3] = R @ u_dofs_global_orig[0:3]
    u_rotated[6:9] = R @ u_dofs_global_orig[6:9]
    u_rotated[3:6] = R @ u_dofs_global_orig[3:6]
    u_rotated[9:12] = R @ u_dofs_global_orig[9:12]
    ele_info_rot = ele_info_orig.copy()
    ele_info_rot['local_z'] = local_z_rot.tolist()
    load_rot = fcn(ele_info_rot, coords_i_rot[0], coords_i_rot[1], coords_i_rot[2], coords_j_rot[0], coords_j_rot[1], coords_j_rot[2], u_rotated)
    assert np.allclose(load_rot, load_orig, atol=1e-10)