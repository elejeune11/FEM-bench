def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    rigid_translation = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, rigid_translation)
    assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    L = 1.0
    axial_disp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0, 0.0])
    shear_disp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0, 0.0, 0.0])
    torsion_disp = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.001, 0.0, 0.0])
    axial_load = fcn(ele_info, xi, yi, zi, xj, yj, zj, axial_disp)
    shear_load = fcn(ele_info, xi, yi, zi, xj, yj, zj, shear_disp)
    torsion_load = fcn(ele_info, xi, yi, zi, xj, yj, zj, torsion_disp)
    expected_axial_force = ele_info['E'] * ele_info['A'] * 0.001 / L
    expected_shear_force = 12 * ele_info['E'] * ele_info['I_z'] * 0.001 / L ** 3
    expected_torsion_moment = ele_info['E'] * ele_info['J'] * 0.001 / (2 * (1 + ele_info['nu']) * L)
    assert abs(axial_load[6] - expected_axial_force) < 1e-06
    assert abs(shear_load[7] - expected_shear_force) < 1e-06
    assert abs(torsion_load[9] - expected_torsion_moment) < 1e-06

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    ua = np.array([0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003, 0.004, 0.005, 0.006, 0.0004, 0.0005, 0.0006])
    ub = np.array([0.0005, 0.0007, 0.0009, 5e-05, 7e-05, 9e-05, 0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003])
    load_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    load_sum = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua) + fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    assert np.allclose(load_combined, load_sum, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs = np.array([0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003, 0.004, 0.005, 0.006, 0.0004, 0.0005, 0.0006])
    load_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    coords_i_rotated = R @ np.array([xi, yi, zi])
    coords_j_rotated = R @ np.array([xj, yj, zj])
    local_z_rotated = R @ ele_info['local_z']
    u_rotated = np.zeros(12)
    for i in range(4):
        start_idx = 3 * i
        end_idx = 3 * (i + 1)
        u_rotated[start_idx:end_idx] = R @ u_dofs[start_idx:end_idx]
    ele_info_rotated = ele_info.copy()
    ele_info_rotated['local_z'] = local_z_rotated
    load_rotated = fcn(ele_info_rotated, *coords_i_rotated, *coords_j_rotated, u_rotated)
    assert np.allclose(load_original, load_rotated, atol=1e-10)