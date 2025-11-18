def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector.
    """
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 2e-05, 'J': 3e-05, 'local_z': np.array([0.0, 1.0, 0.0])}
    (xi, yi, zi) = (1.0, 2.0, 3.0)
    (xj, yj, zj) = (5.0, -1.0, 4.0)
    translation = np.array([0.1, -0.2, 0.3])
    u_dofs_global = np.zeros(12)
    u_dofs_global[0:3] = translation
    u_dofs_global[6:9] = translation
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
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    (E, nu, A) = (210000000000.0, 0.3, 0.01)
    (I_y, I_z, J) = (1e-05, 2e-05, 3e-05)
    G = E / (2 * (1 + nu))
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    u_axial = np.zeros(12)
    u_axial[6] = 1.0
    loads_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    expected_axial = np.zeros(12)
    expected_axial[0] = -E * A / L
    expected_axial[6] = E * A / L
    assert np.allclose(loads_axial, expected_axial)
    u_shear = np.zeros(12)
    u_shear[7] = 1.0
    loads_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    expected_shear = np.zeros(12)
    expected_shear[1] = -12 * E * I_z / L ** 3
    expected_shear[5] = -6 * E * I_z / L ** 2
    expected_shear[7] = 12 * E * I_z / L ** 3
    expected_shear[11] = -6 * E * I_z / L ** 2
    assert np.allclose(loads_shear, expected_shear)
    u_torsion = np.zeros(12)
    u_torsion[9] = 1.0
    loads_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    expected_torsion = np.zeros(12)
    expected_torsion[3] = -G * J / L
    expected_torsion[9] = G * J / L
    assert np.allclose(loads_torsion, expected_torsion)

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds.
    """
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 2e-05, 'J': 3e-05, 'local_z': np.array([0.0, 1.0, 0.0])}
    (xi, yi, zi) = (1.0, 2.0, 3.0)
    (xj, yj, zj) = (5.0, -1.0, 4.0)
    rng = np.random.default_rng(seed=0)
    u_a = rng.random(12)
    u_b = rng.random(12)
    u_combined = u_a + u_b
    f_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_a)
    f_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_b)
    f_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
    assert np.allclose(f_combined, f_a + f_b)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """
    p_i = np.array([1.0, 2.0, 3.0])
    p_j = np.array([5.0, 2.0, 3.0])
    local_z_vec = np.array([0.0, 0.0, 1.0])
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 2e-05, 'J': 3e-05, 'local_z': local_z_vec}
    rng = np.random.default_rng(seed=42)
    u_global = rng.random(12) * 0.001
    f_local_orig = fcn(ele_info, p_i[0], p_i[1], p_i[2], p_j[0], p_j[1], p_j[2], u_global)
    theta = np.pi / 6.0
    (c, s) = (np.cos(theta), np.sin(theta))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    p_i_rot = R @ p_i
    p_j_rot = R @ p_j
    local_z_rot = R @ local_z_vec
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = local_z_rot
    (t1, r1) = (u_global[0:3], u_global[3:6])
    (t2, r2) = (u_global[6:9], u_global[9:12])
    u_global_rot = np.hstack([R @ t1, R @ r1, R @ t2, R @ r2])
    f_local_rot = fcn(ele_info_rot, p_i_rot[0], p_i_rot[1], p_i_rot[2], p_j_rot[0], p_j_rot[1], p_j_rot[2], u_global_rot)
    assert np.allclose(f_local_orig, f_local_rot, atol=1e-09)