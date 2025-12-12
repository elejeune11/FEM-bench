def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 2e-06, 'J': 5e-07, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    u_trans = np.array([0.5, -0.3, 0.2])
    u_dofs_global = np.zeros(12)
    u_dofs_global[0:3] = u_trans
    u_dofs_global[6:9] = u_trans
    load = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global))
    assert load.shape == (12,)
    assert np.allclose(load, np.zeros(12), atol=1e-09, rtol=1e-08)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation
    """
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 2e-06, 'J': 5e-07, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    ua = np.zeros(12)
    ua[0] = 0.0
    ua[6] = 1.0
    fa = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua))
    assert not np.allclose(fa[0], 0.0, atol=1e-12)
    assert np.allclose(fa[0], -fa[6], atol=1e-09, rtol=1e-08)
    assert np.allclose(fa[[1, 2, 3, 4, 5, 7, 8, 9, 10, 11]], 0.0, atol=1e-09, rtol=1e-08)
    ub = np.zeros(12)
    ub[1] = 0.0
    ub[7] = 1.0
    fb = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ub))
    assert not np.allclose(fb[1], 0.0, atol=1e-12)
    assert np.allclose(fb[1], -fb[7], atol=1e-08, rtol=1e-06)
    assert not np.allclose(fb[5], 0.0, atol=1e-12)
    assert np.allclose(fb[5], -fb[11], atol=1e-08, rtol=1e-06)
    assert np.allclose(fb[[0, 2, 3, 4, 6, 8, 9, 10]], 0.0, atol=1e-08, rtol=1e-06)
    uc = np.zeros(12)
    uc[3] = 0.0
    uc[9] = 1.0
    fc = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, uc))
    assert not np.allclose(fc[3], 0.0, atol=1e-12)
    assert np.allclose(fc[3], -fc[9], atol=1e-08, rtol=1e-06)
    assert np.allclose(fc[[0, 1, 2, 4, 5, 6, 7, 8, 10, 11]], 0.0, atol=1e-08, rtol=1e-06)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: f(ua + ub) == f(ua) + f(ub)."""
    ele_info = {'E': 200000000000.0, 'nu': 0.33, 'A': 0.005, 'I_y': 8e-07, 'I_z': 9e-07, 'J': 3e-07, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (-0.5, 0.2, 0.1)
    (xj, yj, zj) = (1.5, -0.3, 0.4)
    ua = np.array([0.1, -0.05, 0.02, 0.01, -0.02, 0.005, -0.03, 0.04, -0.01, 0.002, 0.003, -0.004])
    ub = np.array([-0.02, 0.03, 0.01, -0.005, 0.006, 0.0, 0.015, -0.01, 0.02, -0.001, -0.002, 0.003])
    fa = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua))
    fb = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ub))
    fab = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub))
    assert np.allclose(fab, fa + fb, atol=1e-09, rtol=1e-08)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: rotating the entire configuration (coords, displacements, and local_z) by a global rotation R leaves the local internal end-load vector unchanged."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 2e-06, 'J': 5e-07, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.2, 0.5, -0.3)
    u = np.array([0.05, -0.02, 0.01, 0.01, -0.005, 0.002, -0.03, 0.04, -0.02, -0.006, 0.003, 0.001])
    f0 = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u))
    axis = np.array([1.0, 1.0, 0.5])
    axis = axis / np.linalg.norm(axis)
    theta = 0.7
    (ux, uy, uz) = axis
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c + ux * ux * (1 - c), ux * uy * (1 - c) - uz * s, ux * uz * (1 - c) + uy * s], [uy * ux * (1 - c) + uz * s, c + uy * uy * (1 - c), uy * uz * (1 - c) - ux * s], [uz * ux * (1 - c) - uy * s, uz * uy * (1 - c) + ux * s, c + uz * uz * (1 - c)]])
    ri = R.dot(np.array([xi, yi, zi]))
    rj = R.dot(np.array([xj, yj, zj]))
    local_z_rot = R.dot(np.asarray(ele_info['local_z']))
    u_rot = np.zeros(12)
    ui = u[0:3]
    thetai = u[3:6]
    uj = u[6:9]
    thetaj = u[9:12]
    ui_rot = R.dot(ui)
    uj_rot = R.dot(uj)
    thetai_rot = R.dot(thetai)
    thetaj_rot = R.dot(thetaj)
    u_rot[0:3] = ui_rot
    u_rot[3:6] = thetai_rot
    u_rot[6:9] = uj_rot
    u_rot[9:12] = thetaj_rot
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = local_z_rot
    f_rot = np.asarray(fcn(ele_info_rot, ri[0], ri[1], ri[2], rj[0], rj[1], rj[2], u_rot))
    assert np.allclose(f0, f_rot, atol=1e-08, rtol=1e-06)