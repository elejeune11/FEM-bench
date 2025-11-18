def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    rigid_translation = np.array([0.1, 0.05, -0.02, 0, 0, 0, 0.1, 0.05, -0.02, 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, rigid_translation)
    assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    L = 2.0
    axial_disp = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    axial_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, axial_disp)
    expected_axial_force = ele_info['E'] * ele_info['A'] / L
    assert np.isclose(axial_loads[0], -expected_axial_force)
    assert np.isclose(axial_loads[6], expected_axial_force)
    shear_disp = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    shear_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, shear_disp)
    expected_shear = 12 * ele_info['E'] * ele_info['I_z'] / L ** 3
    assert np.isclose(shear_loads[1], expected_shear)
    assert np.isclose(shear_loads[7], -expected_shear)
    torsion_disp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    torsion_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, torsion_disp)
    G = ele_info['E'] / (2 * (1 + ele_info['nu']))
    expected_torsion = G * ele_info['J'] / L
    assert np.isclose(torsion_loads[3], -expected_torsion)
    assert np.isclose(torsion_loads[9], expected_torsion)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (3, 1, 0)
    ua = np.array([0.001, 0.002, -0.001, 0.0005, -0.0003, 0.0002, 0.002, -0.001, 0.0015, -0.0004, 0.0006, -0.0001])
    ub = np.array([-0.0005, 0.0015, 0.0008, -0.0002, 0.0004, -0.0003, 0.001, 0.002, -0.001, 0.0003, -0.0005, 0.0002])
    u_combined = ua + ub
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    f_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    assert np.allclose(f_combined, fa + fb, rtol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info_original = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    u_original = np.array([0.001, 0.002, -0.001, 0.0005, -0.0003, 0.0002, 0.002, -0.001, 0.0015, -0.0004, 0.0006, -0.0001])
    load_original = fcn(ele_info_original, xi, yi, zi, xj, yj, zj, u_original)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    coords_i_rotated = R @ np.array([xi, yi, zi])
    coords_j_rotated = R @ np.array([xj, yj, zj])
    local_z_rotated = R @ np.array(ele_info_original['local_z'])
    ele_info_rotated = ele_info_original.copy()
    ele_info_rotated['local_z'] = local_z_rotated.tolist()
    u_rotated = np.zeros(12)
    u_rotated[0:3] = R @ u_original[0:3]
    u_rotated[3:6] = R @ u_original[3:6]
    u_rotated[6:9] = R @ u_original[6:9]
    u_rotated[9:12] = R @ u_original[9:12]
    load_rotated = fcn(ele_info_rotated, coords_i_rotated[0], coords_i_rotated[1], coords_i_rotated[2], coords_j_rotated[0], coords_j_rotated[1], coords_j_rotated[2], u_rotated)
    assert np.allclose(load_rotated, load_original, rtol=1e-10)