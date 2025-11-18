def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (2, 0, 0)
    rigid_translation = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0.1, 0.2, 0.3, 0, 0, 0])
    rigid_rotation = np.array([0, 0, 0, 0.05, 0.1, 0.15, 0, 0, 0, 0.05, 0.1, 0.15])
    loads_trans = fcn(ele_info, xi, yi, zi, xj, yj, zj, rigid_translation)
    loads_rot = fcn(ele_info, xi, yi, zi, xj, yj, zj, rigid_rotation)
    assert np.allclose(loads_trans, 0, atol=1e-10)
    assert np.allclose(loads_rot, 0, atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses: (1) Axial unit extension (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    L = 2.0
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (L, 0, 0)
    G = ele_info['E'] / (2 * (1 + ele_info['nu']))
    axial_disp = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    shear_disp = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    torsion_disp = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
    axial_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, axial_disp)
    shear_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, shear_disp)
    torsion_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, torsion_disp)
    expected_axial_force = ele_info['E'] * ele_info['A'] / L
    expected_shear_force = 12 * ele_info['E'] * ele_info['I_z'] / L ** 3
    expected_torsion_moment = G * ele_info['J'] / L
    assert abs(axial_loads[6] - expected_axial_force) < 1e-06
    assert abs(shear_loads[7] - expected_shear_force) < 1e-06
    assert abs(torsion_loads[9] - expected_torsion_moment) < 1e-06

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a combined displacement state (ua + ub) equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (3, 1, 2)
    ua = np.array([0.001, 0.002, 0.003, 0.01, 0.02, 0.03, 0.004, 0.005, 0.006, 0.04, 0.05, 0.06])
    ub = np.array([0.0005, 0.0015, 0.0025, 0.005, 0.015, 0.025, 0.003, 0.004, 0.005, 0.03, 0.04, 0.05])
    loads_ua = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    loads_ub = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    loads_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert np.allclose(loads_combined, loads_ua + loads_ub, rtol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: If we rotate the entire configuration (coords, displacements, and local_z) by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (1, 2, 3)
    (xj, yj, zj) = (4, 5, 6)
    u_dofs = np.array([0.001, 0.002, 0.003, 0.01, 0.02, 0.03, 0.004, 0.005, 0.006, 0.04, 0.05, 0.06])
    baseline_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    coords_i_rot = R @ np.array([xi, yi, zi])
    coords_j_rot = R @ np.array([xj, yj, zj])
    local_z_rot = R @ np.array(ele_info['local_z'])
    u_rotated = np.zeros(12)
    for i in range(0, 12, 3):
        u_rotated[i:i + 3] = R @ u_dofs[i:i + 3]
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = local_z_rot
    rotated_loads = fcn(ele_info_rot, *coords_i_rot, *coords_j_rot, u_rotated)
    assert np.allclose(baseline_loads, rotated_loads, rtol=1e-10)