def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 2e-06, 'J': 5e-07, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.5, -0.3)
    t = np.array([0.123, -0.456, 0.789])
    u_dofs_global = np.zeros(12)
    u_dofs_global[0:3] = t
    u_dofs_global[6:9] = t
    load_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert load_local.shape == (12,)
    assert np.allclose(load_local, np.zeros(12), rtol=1e-08, atol=1e-08)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.02, 'I_y': 3e-06, 'I_z': 4e-06, 'J': 1e-06, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.5, 0.0, 0.0)
    ua = np.zeros(12)
    ua[0] = 0.0
    ua[6] = 1.0
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    assert not np.allclose(fa[0], 0.0, atol=0.0)
    assert not np.allclose(fa[6], 0.0, atol=0.0)
    assert np.allclose(fa[0] + fa[6], 0.0, atol=1e-08)
    assert np.allclose(fa[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11]], np.zeros(10), atol=1e-08)
    ub = np.zeros(12)
    ub[1] = 0.0
    ub[7] = 1.0
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    assert np.allclose(fb[[0, 6, 3, 9]], np.zeros(4), atol=1e-08)
    transverse_related = np.array([1, 4, 7, 10])
    assert np.any(np.abs(fb[transverse_related]) > 1e-10)
    uc = np.zeros(12)
    uc[3] = 0.0
    uc[9] = 1.0
    fc = fcn(ele_info, xi, yi, zi, xj, yj, zj, uc)
    assert not np.allclose(fc[3], 0.0, atol=0.0)
    assert not np.allclose(fc[9], 0.0, atol=0.0)
    assert np.allclose(fc[3] + fc[9], 0.0, atol=1e-08)
    assert np.allclose(fc[[0, 1, 2, 4, 5, 6, 7, 8, 10, 11]], np.zeros(10), atol=1e-08)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a combined displacement state (ua + ub) equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 210000000000.0, 'nu': 0.33, 'A': 0.015, 'I_y': 2.5e-06, 'I_z': 2e-06, 'J': 8e-07, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (-0.5, 0.2, 0.0)
    (xj, yj, zj) = (1.2, -0.3, 0.4)
    np.random.seed(42)
    ua = (np.random.rand(12) - 0.5) * 0.01
    ub = (np.random.rand(12) - 0.5) * 0.01
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert np.allclose(fab, fa + fb, rtol=1e-07, atol=1e-09)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: If we rotate the entire configuration (coords, displacements, and local_z) by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 2e-06, 'J': 5e-07}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.5, -0.3)
    local_z = np.array([0.0, 0.0, 1.0])
    ele_info['local_z'] = local_z.copy()
    u = np.array([0.1, -0.05, 0.02, 0.01, -0.02, 0.03, -0.08, 0.07, -0.04, -0.005, 0.012, -0.02])
    f_orig = fcn(ele_info, xi, yi, zi, xj, yj, zj, u)
    angle = 37.0 * np.pi / 180.0
    axis = np.array([1.0, 1.0, 0.5])
    axis = axis / np.linalg.norm(axis)
    (ux, uy, uz) = axis
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s], [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s], [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)]])
    p_i = np.array([xi, yi, zi])
    p_j = np.array([xj, yj, zj])
    p_i_r = R.dot(p_i)
    p_j_r = R.dot(p_j)
    local_z_r = R.dot(local_z)
    u_rot = np.zeros(12)
    for n in range(2):
        t_idx = n * 6
        trans = u[t_idx:t_idx + 3]
        rot = u[t_idx + 3:t_idx + 6]
        u_rot[t_idx:t_idx + 3] = R.dot(trans)
        u_rot[t_idx + 3:t_idx + 6] = R.dot(rot)
    ele_info_r = ele_info.copy()
    ele_info_r['local_z'] = local_z_r
    f_rot = fcn(ele_info_r, p_i_r[0], p_i_r[1], p_i_r[2], p_j_r[0], p_j_r[1], p_j_r[2], u_rot)
    assert np.allclose(f_orig, f_rot, rtol=1e-07, atol=1e-09)