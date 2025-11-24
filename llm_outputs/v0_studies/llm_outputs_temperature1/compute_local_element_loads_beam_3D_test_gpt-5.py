def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation and rotation of the element produce zero internal forces/moments."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8e-06, 'I_z': 5e-06, 'J': 1.2e-05, 'local_z': np.array([0.0, 1.0, 0.0])}
    (xi, yi, zi) = (1.0, 2.0, 0.5)
    (xj, yj, zj) = (2.2, 3.5, 1.7)
    t = np.array([0.003, -0.002, 0.004])
    r = np.array([0.01, -0.02, 0.03])
    u_dofs_global = np.hstack([t, r, t, r])
    loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert loads.shape == (12,)
    assert np.allclose(loads, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single test covering unit axial extension, transverse y-translation (bending about z), and unit torsion."""
    E = 210000000000.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    A = 0.01
    I_y = 1e-06
    I_z = 2e-06
    J = 3e-06
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[6] = 1.0
    loads_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    k_ax = E * A / L
    assert np.isclose(loads_axial[0], -k_ax, atol=1e-08, rtol=0.0)
    assert np.isclose(loads_axial[6], +k_ax, atol=1e-08, rtol=0.0)
    mask_axial_others = np.ones(12, dtype=bool)
    mask_axial_others[[0, 6]] = False
    assert np.allclose(loads_axial[mask_axial_others], 0.0, atol=1e-08)
    u_bend_y = np.zeros(12)
    u_bend_y[7] = 1.0
    loads_bend_y = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_bend_y)
    k1 = 12.0 * E * I_z / L ** 3
    k2 = 6.0 * E * I_z / L ** 2
    assert np.isclose(loads_bend_y[1], -k1, atol=1e-06, rtol=0.0)
    assert np.isclose(loads_bend_y[5], -k2, atol=1e-06, rtol=0.0)
    assert np.isclose(loads_bend_y[7], +k1, atol=1e-06, rtol=0.0)
    assert np.isclose(loads_bend_y[11], -k2, atol=1e-06, rtol=0.0)
    mask_bend_y_others = np.ones(12, dtype=bool)
    mask_bend_y_others[[1, 5, 7, 11]] = False
    assert np.allclose(loads_bend_y[mask_bend_y_others], 0.0, atol=1e-06)
    u_torsion = np.zeros(12)
    u_torsion[9] = 1.0
    loads_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    k_t = G * J / L
    assert np.isclose(loads_torsion[3], -k_t, atol=1e-08, rtol=0.0)
    assert np.isclose(loads_torsion[9], +k_t, atol=1e-08, rtol=0.0)
    mask_torsion_others = np.ones(12, dtype=bool)
    mask_torsion_others[[3, 9]] = False
    assert np.allclose(loads_torsion[mask_torsion_others], 0.0, atol=1e-08)

def test_superposition_linearity(fcn):
    """Verify linearity: f(ua + ub) = f(ua) + f(ub) for arbitrary displacement states."""
    ele_info = {'E': 70000000000.0, 'nu': 0.25, 'A': 0.005, 'I_y': 4e-06, 'I_z': 6e-06, 'J': 5e-06, 'local_z': np.array([0.2, 0.9, 0.4])}
    (xi, yi, zi) = (1.5, -2.0, 0.75)
    (xj, yj, zj) = (3.2, 1.0, 2.1)
    ua = np.array([0.01, -0.02, 0.005, 0.003, -0.004, 0.002, -0.006, 0.012, -0.009, -0.001, 0.0025, -0.0035])
    ub = np.array([-0.004, 0.008, -0.003, -0.002, 0.0015, -0.0005, 0.007, -0.011, 0.013, 0.0045, -0.003, 0.002])
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert fa.shape == (12,)
    assert fb.shape == (12,)
    assert fab.shape == (12,)
    assert np.allclose(fab, fa + fb, rtol=1e-10, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: rotating coords, local_z, and global DOFs by the same rigid rotation leaves local end-loads unchanged."""

    def rot_x(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def rot_y(b):
        (cb, sb) = (np.cos(b), np.sin(b))
        return np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]])

    def rot_z(c):
        (cc, sc) = (np.cos(c), np.sin(c))
        return np.array([[cc, -sc, 0], [sc, cc, 0], [0, 0, 1]])
    ele_info = {'E': 200000000000.0, 'nu': 0.29, 'A': 0.012, 'I_y': 7.5e-06, 'I_z': 5.5e-06, 'J': 9e-06, 'local_z': np.array([0.1, 0.7, 0.6])}
    (xi, yi, zi) = (1.0, 2.0, 3.0)
    (xj, yj, zj) = (3.0, -1.0, 2.5)
    u = np.array([0.1, 0.2, -0.1, 0.01, -0.02, 0.03, -0.05, 0.15, 0.25, 0.02, -0.01, 0.04])
    loads_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u)
    (a, b, c) = (0.7, -0.4, 0.3)
    R = rot_z(c) @ rot_y(b) @ rot_x(a)
    xi_vec = np.array([xi, yi, zi])
    xj_vec = np.array([xj, yj, zj])
    xi_r = R @ xi_vec
    xj_r = R @ xj_vec
    lz = np.array(ele_info['local_z'])
    lz_r = R @ lz
    T6 = np.zeros((6, 6))
    T6[:3, :3] = R
    T6[3:, 3:] = R
    T = np.zeros((12, 12))
    T[:6, :6] = T6
    T[6:, 6:] = T6
    u_r = T @ u
    ele_info_r = dict(ele_info)
    ele_info_r['local_z'] = lz_r
    loads_local_r = fcn(ele_info_r, xi_r[0], xi_r[1], xi_r[2], xj_r[0], xj_r[1], xj_r[2], u_r)
    assert loads_local.shape == (12,)
    assert loads_local_r.shape == (12,)
    assert np.allclose(loads_local_r, loads_local, rtol=1e-10, atol=1e-10)