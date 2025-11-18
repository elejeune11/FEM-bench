def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (1.0, 2.0, 3.0)
    (xj, yj, zj) = (5.0, 4.0, 3.0)
    u_dofs_global = np.array([0.1, 0.2, 0.3, 0, 0, 0, 0.1, 0.2, 0.3, 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    expected_loads = np.zeros(12)
    assert np.allclose(load_dofs_local, expected_loads, atol=1e-09)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
  (1) Axial unit extension
  (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
  (3) Unit torsional rotation"""
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
    axial_stiffness = E * A / L
    expected_axial[0] = -axial_stiffness
    expected_axial[6] = axial_stiffness
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
    torsional_stiffness = G * J / L
    expected_torsion[3] = -torsional_stiffness
    expected_torsion[9] = torsional_stiffness
    assert np.allclose(load_torsion, expected_torsion)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
combined displacement state (ua + ub) equals the sum of the individual
responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003, 'local_z': [0, 1, 0]}
    (xi, yi, zi) = (1.0, 1.0, 1.0)
    (xj, yj, zj) = (5.0, 4.0, 3.0)
    np.random.seed(0)
    u_a = np.random.rand(12)
    u_b = np.random.rand(12)
    u_c = u_a + u_b
    load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_a)
    load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_b)
    load_c = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_c)
    assert np.allclose(load_c, load_a + load_b)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
If we rotate the entire configuration (coords, displacements, and local_z)
by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003}
    p1 = np.array([1.0, 2.0, 3.0])
    p2 = np.array([5.0, 5.0, 5.0])
    local_z_vec = np.array([0.0, 0.0, 1.0])
    ele_info['local_z'] = local_z_vec
    np.random.seed(42)
    u_global = np.random.rand(12) * 0.01
    load_original = fcn(ele_info, *p1, *p2, u_global)
    theta = np.pi / 4
    (c, s) = (np.cos(theta), np.sin(theta))
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    p1_rot = R @ p1
    p2_rot = R @ p2
    local_z_rot = R @ local_z_vec
    u1_rot = R @ u_global[0:3]
    theta1_rot = R @ u_global[3:6]
    u2_rot = R @ u_global[6:9]
    theta2_rot = R @ u_global[9:12]
    u_global_rot = np.concatenate([u1_rot, theta1_rot, u2_rot, theta2_rot])
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = local_z_rot
    load_rotated = fcn(ele_info_rot, *p1_rot, *p2_rot, u_global_rot)
    assert np.allclose(load_original, load_rotated, atol=1e-09)