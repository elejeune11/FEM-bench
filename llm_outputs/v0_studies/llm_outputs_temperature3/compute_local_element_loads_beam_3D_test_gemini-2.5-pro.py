def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector.
    """
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (1.0, 2.0, 3.0)
    (xj, yj, zj) = (5.0, 4.0, 8.0)
    u_dofs_global = np.array([0.1, -0.2, 0.3, 0.01, -0.02, 0.03, 0.1, -0.2, 0.3, 0.01, -0.02, 0.03])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-09)

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation
    """
    L = 2.0
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_z = 0.0002
    J = 0.0003
    G = E / (2 * (1 + nu))
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': 0.0001, 'I_z': I_z, 'J': J, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[6] = 1.0
    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    expected_axial = np.zeros(12)
    expected_axial[0] = -E * A / L
    expected_axial[6] = E * A / L
    assert np.allclose(load_axial, expected_axial)
    u_shear = np.zeros(12)
    u_shear[7] = 1.0
    load_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    expected_shear = np.zeros(12)
    expected_shear[1] = -12 * E * I_z / L ** 3
    expected_shear[5] = -6 * E * I_z / L ** 2
    expected_shear[7] = 12 * E * I_z / L ** 3
    expected_shear[11] = 6 * E * I_z / L ** 2
    assert np.allclose(load_shear, expected_shear)
    u_torsion = np.zeros(12)
    u_torsion[9] = 1.0
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    expected_torsion = np.zeros(12)
    expected_torsion[3] = -G * J / L
    expected_torsion[9] = G * J / L
    assert np.allclose(load_torsion, expected_torsion)

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds.
    """
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (1.0, 1.0, 1.0)
    (xj, yj, zj) = (2.0, 2.0, 1.0)
    ua = np.random.rand(12) * 0.01
    ub = np.random.rand(12) * 0.01
    u_combined = ua + ub
    load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    load_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    assert np.allclose(load_a + load_b, load_combined)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """
    ele_info_orig = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003, 'local_z': np.array([0.0, 0.0, 1.0])}
    p_i_orig = np.array([1.0, 1.0, 0.0])
    p_j_orig = np.array([3.0, 1.0, 0.0])
    u_dofs_global_orig = np.random.rand(12) * 0.01
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    p_i_rot = R @ p_i_orig
    p_j_rot = R @ p_j_orig
    local_z_rot = R @ ele_info_orig['local_z']
    ele_info_rot = ele_info_orig.copy()
    ele_info_rot['local_z'] = local_z_rot
    Gamma_R = np.kron(np.eye(4, dtype=float), R)
    u_dofs_global_rot = Gamma_R @ u_dofs_global_orig
    load_local_orig = fcn(ele_info_orig, *p_i_orig, *p_j_orig, u_dofs_global_orig)
    load_local_rot = fcn(ele_info_rot, *p_i_rot, *p_j_rot, u_dofs_global_rot)
    assert np.allclose(load_local_orig, load_local_rot, atol=1e-09)