def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    rigid_displacement = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 0.0, 0.0, 0.0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, rigid_displacement)
    assert load_dofs_local.shape == (12,), 'Load vector should have 12 components'
    assert np.allclose(load_dofs_local, 0.0, atol=1e-10), 'Rigid body motion should produce zero internal loads'

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
  (1) Axial unit extension
  (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
  (3) Unit torsional rotation"""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    L = 1.0
    EA_over_L = ele_info['E'] * ele_info['A'] / L
    EI_y_over_L3 = ele_info['E'] * ele_info['I_y'] / L ** 3
    GJ_over_L = ele_info['E'] / (2 * (1 + ele_info['nu'])) * ele_info['J'] / L
    u_axial = np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    assert np.isclose(load_axial[0], EA_over_L, rtol=1e-09), 'Axial force at node i should equal EA/L'
    assert np.isclose(load_axial[6], -EA_over_L, rtol=1e-09), 'Axial force at node j should equal -EA/L'
    u_shear = np.array([0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0, 0.0, 0.0])
    load_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    assert np.isclose(load_shear[1], 12.0 * EI_y_over_L3, rtol=1e-09), 'Transverse shear force at node i should follow beam theory'
    u_torsion = np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, -1.0, 0.0, 0.0])
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    assert np.isclose(load_torsion[3], GJ_over_L, rtol=1e-09), 'Torsional moment at node i should equal GJ/L'
    assert np.isclose(load_torsion[9], -GJ_over_L, rtol=1e-09), 'Torsional moment at node j should equal -GJ/L'

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
combined displacement state (ua + ub) equals the sum of the individual
responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_a = np.array([0.01, 0.002, 0.0, 0.001, 0.0, 0.0, -0.005, 0.001, 0.0, -0.0005, 0.0, 0.0])
    u_b = np.array([0.005, -0.001, 0.0, -0.0005, 0.0, 0.0, 0.002, -0.0015, 0.0, 0.0003, 0.0, 0.0])
    u_combined = u_a + u_b
    load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_a)
    load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_b)
    load_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    load_sum = load_a + load_b
    assert np.allclose(load_combined, load_sum, rtol=1e-09, atol=1e-12), 'Superposition principle should hold: f(ua+ub) = f(ua) + f(ub)'

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
If we rotate the entire configuration (coords, displacements, and local_z)
by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_dofs_global = np.array([0.01, 0.002, 0.003, 0.0005, 0.001, 0.0002, -0.005, 0.001, 0.0015, -0.0003, 0.0005, 0.0001])
    load_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    theta = np.pi / 4
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    R_z = np.array([[cos_t, -sin_t, 0], [sin_t, cos_t, 0], [0, 0, 1]])
    coords_i = np.array([xi, yi, zi])
    coords_j = np.array([xj, yj, zj])
    coords_i_rot = R_z @ coords_i
    coords_j_rot = R_z @ coords_j
    (xi_rot, yi_rot, zi_rot) = coords_i_rot
    (xj_rot, yj_rot, zj_rot) = coords_j_rot
    u_global_reshaped = u_dofs_global.reshape(2, 6)
    u_rot_list = []
    for k in range(2):
        u_trans = u_global_reshaped[k, :3]
        u_rot_trans = R_z @ u_trans
        u_rot_list.extend(u_rot_trans)
        u_rot_list.extend(u_global_reshaped[k, 3:])
    u_dofs_global_rot = np.array(u_rot_list)
    load_rotated = fcn(ele_info, xi_rot, yi_rot, zi_rot, xj_rot, yj_rot, zj_rot, u_dofs_global_rot)
    assert np.allclose(load_original, load_rotated, rtol=1e-09, atol=1e-12), 'Local internal load vector should be invariant under global rigid rotations'