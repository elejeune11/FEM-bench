def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector.
    """
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (1.0, 2.0, 3.0)
    (xj, yj, zj) = (5.0, 4.0, 1.0)
    translation = np.array([0.1, -0.2, 0.05])
    u_dofs_global = np.zeros(12)
    u_dofs_global[0:3] = translation
    u_dofs_global[6:9] = translation
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    expected_loads = np.zeros(12)
    assert np.allclose(load_dofs_local, expected_loads, atol=1e-09)

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation
    """
    L = 2.0
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 2e-05, 'J': 3e-05, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    (E, nu, A, I_z, J) = (ele_info['E'], ele_info['nu'], ele_info['A'], ele_info['I_z'], ele_info['J'])
    G = E / (2 * (1 + nu))
    u_axial = np.array([0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0, 0])
    loads_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    expected_axial = np.zeros(12)
    expected_axial[0] = -E * A / L
    expected_axial[6] = E * A / L
    assert np.allclose(loads_axial, expected_axial)
    u_shear_y = np.array([0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0, 0, 0])
    loads_shear_y = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y)
    expected_shear_y = np.zeros(12)
    expected_shear_y[1] = -12 * E * I_z / L ** 3
    expected_shear_y[5] = -6 * E * I_z / L ** 2
    expected_shear_y[7] = 12 * E * I_z / L ** 3
    expected_shear_y[11] = 6 * E * I_z / L ** 2
    assert np.allclose(loads_shear_y, expected_shear_y)
    u_torsion = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0, 0, 0])
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
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 1.0, 0.0])}
    (xi, yi, zi) = (1.0, 1.0, 1.0)
    (xj, yj, zj) = (5.0, 4.0, 3.0)
    np.random.seed(0)
    u_a = np.random.rand(12) * 0.01
    u_b = np.random.rand(12) * 0.01
    u_c = u_a + u_b
    load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_a)
    load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_b)
    load_c = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_c)
    assert np.allclose(load_c, load_a + load_b)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """
    ele_info_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.02, 'I_y': 0.00015, 'I_z': 0.00025, 'J': 0.0004}
    L = 5.0
    p1_init = np.array([1.0, 1.0, 1.0])
    p2_init = np.array([1.0 + L, 1.0, 1.0])
    local_z_init = np.array([0.0, 0.0, 1.0])
    ele_info_init = ele_info_props.copy()
    ele_info_init['local_z'] = local_z_init
    u_dofs_init = np.array([0, 0, 0, 0, 0, 0, 0, 0.1, 0.05, 0.01, 0.02, 0.03])
    theta = np.pi / 6
    axis = np.array([1, 2, 3]) / np.sqrt(14)
    (c, s) = (np.cos(theta), np.sin(theta))
    t = 1 - c
    (x, y, z) = axis
    R = np.array([[t * x * x + c, t * x * y - s * z, t * x * z + s * y], [t * x * y + s * z, t * y * y + c, t * y * z - s * x], [t * x * z - s * y, t * y * z + s * x, t * z * z + c]])
    p1_rot = R @ p1_init
    p2_rot = R @ p2_init
    local_z_rot = R @ local_z_init
    ele_info_rot = ele_info_props.copy()
    ele_info_rot['local_z'] = local_z_rot
    T_block = block_diag(R, R, R, R)
    u_dofs_rot = T_block @ u_dofs_init
    load_local_init = fcn(ele_info_init, *p1_init, *p2_init, u_dofs_init)
    load_local_rot = fcn(ele_info_rot, *p1_rot, *p2_rot, u_dofs_rot)
    assert np.allclose(load_local_init, load_local_rot, atol=1e-09)