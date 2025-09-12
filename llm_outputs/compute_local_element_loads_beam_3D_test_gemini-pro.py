def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 2e-05, 'J': 5e-06}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    translation = np.array([1.0, 2.0, 3.0])
    u_dofs_global = np.concatenate([translation, [0, 0, 0], translation, [0, 0, 0]])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert_allclose(load_dofs_local, np.zeros(12))

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 2e-05, 'J': 5e-06}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    L = 1.0
    u_dofs_global = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert_allclose(load_dofs_local[0], ele_info['E'] * ele_info['A'] / L)
    assert_allclose(load_dofs_local[6], -ele_info['E'] * ele_info['A'] / L)
    u_dofs_global = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert_allclose(load_dofs_local[1], 12 * ele_info['E'] * ele_info['I_z'] / L ** 3)
    assert_allclose(load_dofs_local[7], -12 * ele_info['E'] * ele_info['I_z'] / L ** 3)
    u_dofs_global = np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert_allclose(load_dofs_local[3], ele_info['J'] * ele_info['E'] / L)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 2e-05, 'J': 5e-06}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    ua = np.random.rand(12)
    ub = np.random.rand(12)
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert_allclose(fab, fa + fb)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 2e-05, 'J': 5e-06, 'local_z': [0, 0, 1]}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.random.rand(12)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    p_i = np.array([xi, yi, zi])
    p_j = np.array([xj, yj, zj])
    p_i_rotated = R @ p_i
    p_j_rotated = R @ p_j
    local_z_rotated = R @ ele_info['local_z']
    ele_info_rotated = ele_info.copy()
    ele_info_rotated['local_z'] = local_z_rotated
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    load_dofs_local_rotated = fcn(ele_info_rotated, *p_i_rotated, *p_j_rotated, u_dofs_global)
    assert_allclose(load_dofs_local, load_dofs_local_rotated)