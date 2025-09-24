def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    rigid_translation = [0.1, 0.05, -0.02, 0.0, 0.0, 0.0, 0.1, 0.05, -0.02, 0.0, 0.0, 0.0]
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, rigid_translation)
    assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-12)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    L = 1.0
    u_axial = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    expected_axial_force = ele_info['E'] * ele_info['A'] / L
    assert np.abs(load_axial[0] + expected_axial_force) < 1e-06
    assert np.abs(load_axial[6] - expected_axial_force) < 1e-06
    u_shear = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    load_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    expected_shear_force = 12 * ele_info['E'] * ele_info['I_z'] / L ** 3
    assert np.abs(load_shear[1] + expected_shear_force) < 1e-06
    assert np.abs(load_shear[7] - expected_shear_force) < 1e-06
    u_torsion = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0]
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    G = ele_info['E'] / (2 * (1 + ele_info['nu']))
    expected_torque = G * ele_info['J'] / L
    assert np.abs(load_torsion[3] + expected_torque) < 1e-06
    assert np.abs(load_torsion[9] - expected_torque) < 1e-06

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': None}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 1.0, 0.5)
    ua = [0.01, 0.02, -0.005, 0.001, -0.002, 0.003, 0.015, -0.01, 0.008, -0.001, 0.004, -0.002]
    ub = [-0.005, 0.01, 0.003, -0.002, 0.001, -0.001, 0.02, 0.005, -0.01, 0.003, -0.001, 0.002]
    u_combined = [ua[i] + ub[i] for i in range(12)]
    load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    load_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    load_sum = load_a + load_b
    assert np.allclose(load_combined, load_sum, rtol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-05, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 1.0, 0.5)
    u_dofs = [0.01, 0.02, -0.005, 0.001, -0.002, 0.003, 0.015, -0.01, 0.008, -0.001, 0.004, -0.002]
    load_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    pos_i = R @ np.array([xi, yi, zi])
    pos_j = R @ np.array([xj, yj, zj])
    (xi_rot, yi_rot, zi_rot) = pos_i
    (xj_rot, yj_rot, zj_rot) = pos_j
    local_z_rot = R @ ele_info['local_z']
    R_12 = np.zeros((12, 12))
    for i in range(4):
        R_12[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
    u_dofs_rot = R_12 @ np.array(u_dofs)
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = local_z_rot
    load_rotated = fcn(ele_info_rot, xi_rot, yi_rot, zi_rot, xj_rot, yj_rot, zj_rot, u_dofs_rot)
    assert np.allclose(load_original, load_rotated, rtol=1e-10)