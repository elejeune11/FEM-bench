def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 2e-06
    I_z = 3e-06
    J = 5e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (2.5, 0.0, 0.0)
    t = np.array([0.5, -1.2, 3.4])
    u_dofs_global = np.array([t[0], t[1], t[2], 0.0, 0.0, 0.0, t[0], t[1], t[2], 0.0, 0.0, 0.0], dtype=float)
    loads_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert loads_local.shape == (12,)
    assert np.allclose(loads_local, 0.0, atol=1e-12, rtol=0.0)

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation
    """
    E = 210000000000.0
    nu = 0.29
    G = E / (2.0 * (1.0 + nu))
    A = 0.01
    I_y = 2e-06
    I_z = 3e-06
    J = 5e-06
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    f_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    EA_over_L = E * A / L
    assert np.allclose(f_axial[0], -EA_over_L, rtol=1e-12, atol=1e-09)
    assert np.allclose(f_axial[6], +EA_over_L, rtol=1e-12, atol=1e-09)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[0, 6]] = False
    assert np.allclose(f_axial[mask_other], 0.0, atol=1e-09, rtol=0.0)
    u_shear_y = np.zeros(12)
    u_shear_y[1] = 0.0
    u_shear_y[7] = 1.0
    f_shear_y = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y)
    EIz = E * I_z
    Fy_i = f_shear_y[1]
    Fy_j = f_shear_y[7]
    Mz_i = f_shear_y[5]
    Mz_j = f_shear_y[11]
    Fy_expected = 12.0 * EIz / L ** 3
    Mz_expected = 6.0 * EIz / L ** 2
    assert np.allclose(Fy_i, -Fy_expected, rtol=1e-12, atol=1e-09)
    assert np.allclose(Fy_j, +Fy_expected, rtol=1e-12, atol=1e-09)
    assert np.allclose(Mz_i, -Mz_expected, rtol=1e-12, atol=1e-09)
    assert np.allclose(Mz_j, +Mz_expected, rtol=1e-12, atol=1e-09)
    assert np.allclose(Fy_i + Fy_j, 0.0, atol=1e-09, rtol=0.0)
    assert np.allclose(Mz_i + Mz_j, 0.0, atol=1e-09, rtol=0.0)
    assert np.allclose(Fy_i, 2.0 / L * Mz_i, rtol=1e-12, atol=1e-09)
    assert np.allclose(Fy_j, 2.0 / L * Mz_j, rtol=1e-12, atol=1e-09)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[1, 5, 7, 11]] = False
    assert np.allclose(f_shear_y[mask_other], 0.0, atol=1e-09, rtol=0.0)
    u_torsion = np.zeros(12)
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    f_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    GJ_over_L = G * J / L
    assert np.allclose(f_torsion[3], -GJ_over_L, rtol=1e-12, atol=1e-09)
    assert np.allclose(f_torsion[9], +GJ_over_L, rtol=1e-12, atol=1e-09)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[3, 9]] = False
    assert np.allclose(f_torsion[mask_other], 0.0, atol=1e-09, rtol=0.0)

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: f(ua + ub) == f(ua) + f(ub).
    """
    E = 70000000000.0
    nu = 0.33
    A = 0.02
    I_y = 1.5e-06
    I_z = 2.5e-06
    J = 4e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (3.0, 0.0, 0.0)
    rng = np.random.default_rng(12345)
    ua = rng.normal(size=12)
    ub = rng.normal(size=12)
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert fa.shape == (12,)
    assert fb.shape == (12,)
    assert fab.shape == (12,)
    assert np.allclose(fab, fa + fb, rtol=1e-12, atol=1e-09)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance: rotating the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R should leave the local internal end-load vector unchanged.
    """
    E = 200000000000.0
    nu = 0.28
    A = 0.015
    I_y = 3e-06
    I_z = 4e-06
    J = 6e-06
    ele_info_base = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.5, -0.7, 1.2)
    L = 2.7
    xj, yj, zj = (xi + L, yi, zi)
    u_base = np.array([0.3, -0.1, 0.2, 0.05, -0.02, 0.03, -0.4, 0.25, -0.15, -0.01, 0.06, -0.04], dtype=float)
    f_base = fcn(ele_info_base, xi, yi, zi, xj, yj, zj, u_base)
    theta = 0.7
    c, s = (np.cos(theta), np.sin(theta))
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    Pi = np.array([xi, yi, zi])
    Pj = np.array([xj, yj, zj])
    Pi_r = R @ Pi
    Pj_r = R @ Pj
    xi_r, yi_r, zi_r = Pi_r.tolist()
    xj_r, yj_r, zj_r = Pj_r.tolist()
    ele_info_rot = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': R @ ele_info_base['local_z']}
    u1_t = np.array(u_base[0:3])
    u1_r = np.array(u_base[3:6])
    u2_t = np.array(u_base[6:9])
    u2_r = np.array(u_base[9:12])
    u1_t_r = R @ u1_t
    u1_r_r = R @ u1_r
    u2_t_r = R @ u2_t
    u2_r_r = R @ u2_r
    u_rot = np.zeros(12, dtype=float)
    u_rot[0:3] = u1_t_r
    u_rot[3:6] = u1_r_r
    u_rot[6:9] = u2_t_r
    u_rot[9:12] = u2_r_r
    f_rot = fcn(ele_info_rot, xi_r, yi_r, zi_r, xj_r, yj_r, zj_r, u_rot)
    assert f_base.shape == (12,)
    assert f_rot.shape == (12,)
    assert np.allclose(f_rot, f_base, rtol=1e-12, atol=1e-09)