def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    u_dofs_global = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0.1, 0.2, 0.3, 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
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
    G = ele_info['E'] / (2 * (1 + ele_info['nu']))
    u_torsion = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    expected_torsion_moment = G * ele_info['J'] / L
    assert np.isclose(load_torsion[3], expected_torsion_moment)
    assert np.isclose(load_torsion[9], -expected_torsion_moment)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (3, 0, 0)
    ua = np.array([0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003, 0.0015, 0.0025, 0.0035, 0.00015, 0.00025, 0.00035])
    ub = np.array([0.0005, 0.001, 0.0015, 5e-05, 0.0001, 0.00015, 0.0008, 0.0012, 0.0018, 8e-05, 0.00012, 0.00018])
    u_combined = ua + ub
    load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    load_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    assert np.allclose(load_combined, load_a + load_b, rtol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    u_dofs_global = np.array([0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003, 0.0015, 0.0025, 0.0035, 0.00015, 0.00025, 0.00035])
    load_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    coords_i_rot = R @ np.array([xi, yi, zi])
    coords_j_rot = R @ np.array([xj, yj, zj])
    local_z_rot = R @ np.array(ele_info['local_z'])
    u_rotated = np.zeros(12)
    for i in range(0, 12, 6):
        u_rotated[i:i + 3] = R @ u_dofs_global[i:i + 3]
        u_rotated[i + 3:i + 6] = R @ u_dofs_global[i + 3:i + 6]
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = local_z_rot.tolist()
    load_rotated = fcn(ele_info_rot, *coords_i_rot, *coords_j_rot, u_rotated)
    assert np.allclose(load_original, load_rotated, rtol=1e-10)