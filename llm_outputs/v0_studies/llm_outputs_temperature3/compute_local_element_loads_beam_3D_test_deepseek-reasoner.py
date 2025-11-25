def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    rigid_translation = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0.1, 0.2, 0.3, 0, 0, 0])
    rigid_rotation_x = np.array([0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.1, 0, 0])
    rigid_rotation_y = np.array([0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.1, 0])
    rigid_rotation_z = np.array([0, 0, 0, 0, 0, 0.1, 0, 0, 0, 0, 0, 0.1])
    for displacement in [rigid_translation, rigid_rotation_x, rigid_rotation_y, rigid_rotation_z]:
        load_vector = fcn(ele_info, xi, yi, zi, xj, yj, zj, displacement)
        assert np.allclose(load_vector, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    L = 2.0
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (L, 0, 0)
    axial_disp = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    axial_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, axial_disp)
    expected_axial_force = ele_info['E'] * ele_info['A'] / L
    assert abs(axial_loads[6] + expected_axial_force) < 1e-06
    shear_disp = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    shear_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, shear_disp)
    assert abs(shear_loads[1] + shear_loads[7]) < 1e-10
    assert abs(shear_loads[5] - (shear_loads[11] + shear_loads[1] * L)) < 1e-10
    torsion_disp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    torsion_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, torsion_disp)
    G = ele_info['E'] / (2 * (1 + ele_info['nu']))
    expected_torque = G * ele_info['J'] / L
    assert abs(torsion_loads[9] + expected_torque) < 1e-06

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (3, 0, 0)
    ua = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012])
    ub = np.array([0.0005, 0.0015, 0.0025, 0.0035, 0.0045, 0.0055, 0.0065, 0.0075, 0.0085, 0.0095, 0.0105, 0.0115])
    u_combined = ua + ub
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    f_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    assert np.allclose(f_combined, fa + fb, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info_original = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi1, yi1, zi1) = (0, 0, 0)
    (xj1, yj1, zj1) = (2, 0, 0)
    u_original = np.array([0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01, 0.011, 0.012])
    loads_original = fcn(ele_info_original, xi1, yi1, zi1, xj1, yj1, zj1, u_original)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    coords_i_rotated = R @ np.array([xi1, yi1, zi1])
    coords_j_rotated = R @ np.array([xj1, yj1, zj1])
    local_z_rotated = R @ np.array([0, 0, 1])
    u_rotated = np.zeros(12)
    u_rotated[0:3] = R @ u_original[0:3]
    u_rotated[3:6] = R @ u_original[3:6]
    u_rotated[6:9] = R @ u_original[6:9]
    u_rotated[9:12] = R @ u_original[9:12]
    ele_info_rotated = ele_info_original.copy()
    ele_info_rotated['local_z'] = local_z_rotated.tolist()
    loads_rotated = fcn(ele_info_rotated, coords_i_rotated[0], coords_i_rotated[1], coords_i_rotated[2], coords_j_rotated[0], coords_j_rotated[1], coords_j_rotated[2], u_rotated)
    assert np.allclose(loads_original, loads_rotated, atol=1e-10)