def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8e-06, 'I_z': 5e-06, 'J': 1e-05, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    t = np.array([0.2, -1.5, 0.7])
    (u1, v1, w1) = t
    (u2, v2, w2) = t
    u_dofs_global = np.array([u1, v1, w1, 0.0, 0.0, 0.0, u2, v2, w2, 0.0, 0.0, 0.0], dtype=float)
    loads_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(loads_local, np.zeros(12), rtol=0.0, atol=1e-12)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation"""
    E = 210000000000.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    A = 0.01
    I_y = 8e-06
    I_z = 5e-06
    J = 1e-05
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    L = np.linalg.norm([xj - xi, yj - yi, zj - zi])
    K = np.zeros((12, 12), dtype=float)
    k_ax = E * A / L * np.array([[1, -1], [-1, 1]], dtype=float)
    K[np.ix_([0, 6], [0, 6])] = k_ax
    k_tor = G * J / L * np.array([[1, -1], [-1, 1]], dtype=float)
    K[np.ix_([3, 9], [3, 9])] = k_tor
    k_bz = E * I_z / L ** 3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]], dtype=float)
    idx_bz = [1, 5, 7, 11]
    K[np.ix_(idx_bz, idx_bz)] = k_bz
    k_by = E * I_y / L ** 3 * np.array([[12, 6 * L, -12, 6 * L], [6 * L, 4 * L ** 2, -6 * L, 2 * L ** 2], [-12, -6 * L, 12, -6 * L], [6 * L, 2 * L ** 2, -6 * L, 4 * L ** 2]], dtype=float)
    idx_by = [2, 4, 8, 10]
    K[np.ix_(idx_by, idx_by)] = k_by
    U1 = np.zeros(12)
    U1[6] = 1.0
    expected1 = K @ U1
    loads1 = fcn(ele_info, xi, yi, zi, xj, yj, zj, U1)
    assert np.allclose(loads1, expected1, rtol=1e-10, atol=1e-12)
    U2 = np.zeros(12)
    U2[7] = 1.0
    expected2 = K @ U2
    loads2 = fcn(ele_info, xi, yi, zi, xj, yj, zj, U2)
    assert np.allclose(loads2, expected2, rtol=1e-10, atol=1e-12)
    U3 = np.zeros(12)
    U3[9] = 1.0
    expected3 = K @ U3
    loads3 = fcn(ele_info, xi, yi, zi, xj, yj, zj, U3)
    assert np.allclose(loads3, expected3, rtol=1e-10, atol=1e-12)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: f(ua + ub) == f(ua) + f(ub) for arbitrary displacement states."""
    ele_info = {'E': 200000000000.0, 'nu': 0.29, 'A': 0.012, 'I_y': 7.5e-06, 'I_z': 6.2e-06, 'J': 9.1e-06, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.3, -1.2, 0.5)
    (xj, yj, zj) = (2.1, 0.8, 1.4)
    ua = np.array([0.05, -0.02, 0.03, 0.004, -0.006, 0.007, -0.01, 0.015, -0.02, 0.002, -0.003, 0.001], dtype=float)
    ub = np.array([-0.03, 0.04, -0.01, -0.005, 0.002, -0.004, 0.02, -0.01, 0.005, -0.001, 0.006, -0.002], dtype=float)
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert np.allclose(fab, fa + fb, rtol=1e-10, atol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: rotating the geometry, displacements, and local_z by a global rotation leaves the local end-load vector unchanged."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 9e-06, 'I_z': 7e-06, 'J': 1.2e-05, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    U = np.array([0.1, -0.05, 0.02, 0.01, -0.02, 0.03, -0.08, 0.07, -0.04, -0.02, 0.05, -0.01], dtype=float)
    angle = np.deg2rad(33.0)
    (c, s) = (np.cos(angle), np.sin(angle))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    ri = np.array([xi, yi, zi])
    rj = np.array([xj, yj, zj])
    ri_rot = R @ ri
    rj_rot = R @ rj
    (xi_r, yi_r, zi_r) = ri_rot.tolist()
    (xj_r, yj_r, zj_r) = rj_rot.tolist()
    ele_info_rot = dict(ele_info)
    ele_info_rot['local_z'] = R @ ele_info['local_z']
    U_rot = np.zeros_like(U)
    U_rot[0:3] = R @ U[0:3]
    U_rot[3:6] = R @ U[3:6]
    U_rot[6:9] = R @ U[6:9]
    U_rot[9:12] = R @ U[9:12]
    f_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, U)
    f_local_rot = fcn(ele_info_rot, xi_r, yi_r, zi_r, xj_r, yj_r, zj_r, U_rot)
    assert np.allclose(f_local, f_local_rot, rtol=1e-10, atol=1e-12)