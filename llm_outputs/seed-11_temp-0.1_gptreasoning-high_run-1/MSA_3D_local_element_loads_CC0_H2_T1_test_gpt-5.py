def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-06
    I_z = 1e-06
    J = 2e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    L = 2.0
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    t = np.array([0.13, -0.271, 0.032], dtype=float)
    u_dofs = np.zeros(12, dtype=float)
    u_dofs[0:3] = t
    u_dofs[3:6] = 0.0
    u_dofs[6:9] = t
    u_dofs[9:12] = 0.0
    f_local = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs))
    assert f_local.shape == (12,)
    assert np.allclose(f_local, np.zeros(12), atol=1e-09, rtol=0.0)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation
    """
    E = 210000000000.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    A = 0.01
    I_y = 1e-06
    I_z = 1e-06
    J = 2e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    L = 2.0
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    u_axial = np.zeros(12, dtype=float)
    u_axial[6] = 1.0
    f_axial = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial))
    N = E * A / L
    assert np.isclose(f_axial[0] + f_axial[6], 0.0, atol=1e-08)
    assert np.isclose(abs(f_axial[0]), N, rtol=1e-12, atol=1e-08)
    mask = np.ones(12, dtype=bool)
    mask[[0, 6]] = False
    assert np.allclose(f_axial[mask], 0.0, atol=1e-08)
    u_shear = np.zeros(12, dtype=float)
    u_shear[7] = 1.0
    f_shear = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear))
    Vy = 12.0 * E * I_z / L ** 3
    Mz = 6.0 * E * I_z / L ** 2
    assert np.isclose(f_shear[1] + f_shear[7], 0.0, atol=1e-06)
    assert np.isclose(abs(f_shear[1]), Vy, rtol=1e-12, atol=1e-06)
    assert np.isclose(f_shear[5], f_shear[11], atol=1e-06)
    assert np.isclose(abs(f_shear[5]), Mz, rtol=1e-12, atol=1e-06)
    mask = np.ones(12, dtype=bool)
    mask[[1, 5, 7, 11]] = False
    assert np.allclose(f_shear[mask], 0.0, atol=1e-06)
    u_torsion = np.zeros(12, dtype=float)
    u_torsion[9] = 1.0
    f_torsion = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion))
    Tx = G * J / L
    assert np.isclose(f_torsion[3] + f_torsion[9], 0.0, atol=1e-08)
    assert np.isclose(abs(f_torsion[3]), Tx, rtol=1e-12, atol=1e-08)
    mask = np.ones(12, dtype=bool)
    mask[[3, 9]] = False
    assert np.allclose(f_torsion[mask], 0.0, atol=1e-08)

def test_superposition_linearity(fcn):
    """Verify linearity: f(ua + ub) == f(ua) + f(ub) for arbitrary displacement states."""
    E = 70000000000.0
    nu = 0.33
    A = 0.005
    I_y = 2e-06
    I_z = 1.5e-06
    J = 3e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (1.3, -0.4, 2.2)
    xj, yj, zj = (2.7, 1.6, -0.8)
    rng = np.random.default_rng(42)
    ua = rng.standard_normal(12)
    ub = rng.standard_normal(12)
    fa = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua))
    fb = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ub))
    fab = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub))
    assert fa.shape == (12,)
    assert fb.shape == (12,)
    assert fab.shape == (12,)
    assert np.allclose(fab, fa + fb, rtol=1e-10, atol=1e-09)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: rotating coordinates, displacements, and local_z by a rigid global rotation leaves the local internal end-load vector unchanged."""
    E = 200000000000.0
    nu = 0.28
    A = 0.007
    I_y = 2.5e-06
    I_z = 1.2e-06
    J = 3.3e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (1.2, -0.7, 0.5)
    xj, yj, zj = (3.8, 2.1, -1.0)
    u = np.zeros(12, dtype=float)
    u[0:3] = np.array([0.05, -0.02, 0.01])
    u[3:6] = np.array([0.01, -0.03, 0.02])
    u[6:9] = np.array([0.02, 0.04, -0.05])
    u[9:12] = np.array([-0.02, 0.01, 0.03])
    f_base = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u))

    def Rx(a):
        c, s = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)

    def Ry(a):
        c, s = (np.cos(a), np.sin(a))
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)

    def Rz(a):
        c, s = (np.cos(a), np.sin(a))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)
    R = Rz(0.6) @ Ry(-0.4) @ Rx(0.2)
    pi = np.array([xi, yi, zi], dtype=float)
    pj = np.array([xj, yj, zj], dtype=float)
    pi_r = R @ pi
    pj_r = R @ pj
    zloc = np.array(ele_info['local_z'], dtype=float)
    zloc_r = R @ zloc
    u_r = np.zeros(12, dtype=float)
    u_r[0:3] = R @ u[0:3]
    u_r[3:6] = R @ u[3:6]
    u_r[6:9] = R @ u[6:9]
    u_r[9:12] = R @ u[9:12]
    ele_info_rot = dict(ele_info)
    ele_info_rot['local_z'] = zloc_r
    f_rot = np.asarray(fcn(ele_info_rot, pi_r[0], pi_r[1], pi_r[2], pj_r[0], pj_r[1], pj_r[2], u_r))
    assert f_base.shape == (12,)
    assert f_rot.shape == (12,)
    assert np.allclose(f_rot, f_base, rtol=1e-10, atol=1e-09)