def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': None}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0.1, 0.2, 0.3, 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': None}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
    u_axial = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    expected_axial_force = ele_info['E'] * ele_info['A'] / L
    assert np.isclose(load_axial[6], -expected_axial_force)
    assert np.isclose(load_axial[0], expected_axial_force)
    u_shear_y = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    load_shear_y = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y)
    expected_shear_force = 12 * ele_info['E'] * ele_info['I_z'] / L ** 3
    assert np.isclose(load_shear_y[7], -expected_shear_force)
    assert np.isclose(load_shear_y[1], expected_shear_force)
    u_torsion = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0])
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    expected_torsion_moment = ele_info['E'] * ele_info['J'] / (2 * (1 + ele_info['nu']) * L)
    assert np.isclose(load_torsion[3], -expected_torsion_moment)
    assert np.isclose(load_torsion[9], expected_torsion_moment)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': None}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    ua = np.array([0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003, 0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003])
    ub = np.array([0.002, 0.001, 0.004, 0.0002, 0.0001, 0.0004, 0.002, 0.001, 0.004, 0.0002, 0.0001, 0.0004])
    load_ua = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    load_ub = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    load_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert np.allclose(load_combined, load_ua + load_ub, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.array([0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003, 0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003])
    load_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    coords_i_rotated = R @ np.array([xi, yi, zi])
    coords_j_rotated = R @ np.array([xj, yj, zj])
    local_z_rotated = R @ ele_info['local_z']
    u_rotated = np.zeros(12)
    for i in range(4):
        u_rotated[3 * i:3 * i + 3] = R @ u_dofs_global[3 * i:3 * i + 3]
        u_rotated[3 * i + 3:3 * i + 6] = R @ u_dofs_global[3 * i + 3:3 * i + 6]
    ele_info_rotated = ele_info.copy()
    ele_info_rotated['local_z'] = local_z_rotated
    load_rotated = fcn(ele_info_rotated, coords_i_rotated[0], coords_i_rotated[1], coords_i_rotated[2], coords_j_rotated[0], coords_j_rotated[1], coords_j_rotated[2], u_rotated)
    assert np.allclose(load_rotated, load_original, atol=1e-10)