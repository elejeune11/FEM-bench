def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    rigid_translation = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0.1, 0.2, 0.3, 0, 0, 0])
    local_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, rigid_translation)
    assert np.allclose(local_loads, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.67e-06
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (L, 0, 0)
    axial_disp = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    axial_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, axial_disp)
    expected_axial_force = E * A / L
    assert np.isclose(axial_loads[6], -expected_axial_force)
    assert np.isclose(axial_loads[0], expected_axial_force)
    shear_disp = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    shear_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, shear_disp)
    expected_shear_force = 12 * E * I_z / L ** 3
    assert np.isclose(shear_loads[7], -expected_shear_force)
    assert np.isclose(shear_loads[1], expected_shear_force)
    torsion_disp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    torsion_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, torsion_disp)
    G = E / (2 * (1 + nu))
    expected_torsion_moment = G * J / L
    assert np.isclose(torsion_loads[9], -expected_torsion_moment)
    assert np.isclose(torsion_loads[3], expected_torsion_moment)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (3, 1, 2)
    ua = np.array([0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003, 0.004, 0.005, 0.006, 0.0004, 0.0005, 0.0006])
    ub = np.array([0.0005, -0.001, 0.002, -0.0002, 0.0003, -0.0001, 0.003, -0.002, 0.004, 0.0001, -0.0004, 0.0002])
    u_combined = ua + ub
    loads_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    loads_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    loads_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    assert np.allclose(loads_combined, loads_a + loads_b, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info_original = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (1, 2, 3)
    (xj, yj, zj) = (4, 5, 6)
    u_original = np.array([0.001, 0.002, 0.003, 0.0001, 0.0002, 0.0003, 0.004, 0.005, 0.006, 0.0004, 0.0005, 0.0006])
    loads_original = fcn(ele_info_original, xi, yi, zi, xj, yj, zj, u_original)
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    coords_i_rotated = R @ np.array([xi, yi, zi])
    coords_j_rotated = R @ np.array([xj, yj, zj])
    local_z_rotated = R @ np.array([0, 0, 1])
    u_rotated = np.zeros(12)
    for i in range(0, 12, 6):
        u_rotated[i:i + 3] = R @ u_original[i:i + 3]
        u_rotated[i + 3:i + 6] = R @ u_original[i + 3:i + 6]
    ele_info_rotated = ele_info_original.copy()
    ele_info_rotated['local_z'] = local_z_rotated.tolist()
    loads_rotated = fcn(ele_info_rotated, coords_i_rotated[0], coords_i_rotated[1], coords_i_rotated[2], coords_j_rotated[0], coords_j_rotated[1], coords_j_rotated[2], u_rotated)
    assert np.allclose(loads_rotated, loads_original, atol=1e-10)