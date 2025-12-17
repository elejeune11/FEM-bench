def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments
    in the local load vector.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.02
    I_y = 8e-06
    I_z = 5e-06
    J = 1.2e-05
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (2.0, 0.0, 0.0)
    t = np.array([0.003, -0.002, 0.004], dtype=float)
    r = np.array([0.0, 0.0, 0.0], dtype=float)
    u_dofs_global = np.hstack([t, r, t, r])
    loads_local = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global))
    assert loads_local.shape == (12,)
    assert np.allclose(loads_local, np.zeros(12), atol=1e-09)

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation
    """
    E = 210000000000.0
    nu = 0.25
    A = 0.03
    I_y = 3e-06
    I_z = 4e-06
    J = 5e-06
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    loads_axial = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial))
    EA_over_L = E * A / L
    exp_axial = np.zeros(12)
    exp_axial[0] = -EA_over_L
    exp_axial[6] = EA_over_L
    assert np.allclose(loads_axial, exp_axial, rtol=1e-12, atol=1e-08)
    u_shear_y = np.zeros(12)
    u_shear_y[1] = 0.0
    u_shear_y[7] = 1.0
    loads_shear_y = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y))
    k_bz = E * I_z / L ** 3
    exp_shear_y = np.zeros(12)
    exp_shear_y[1] = -12.0 * k_bz
    exp_shear_y[5] = -6.0 * L * k_bz
    exp_shear_y[7] = 12.0 * k_bz
    exp_shear_y[11] = -6.0 * L * k_bz
    assert np.allclose(loads_shear_y, exp_shear_y, rtol=1e-12, atol=1e-08)
    u_torsion = np.zeros(12)
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    loads_torsion = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion))
    G = E / (2.0 * (1.0 + nu))
    GJ_over_L = G * J / L
    exp_torsion = np.zeros(12)
    exp_torsion[3] = -GJ_over_L
    exp_torsion[9] = GJ_over_L
    assert np.allclose(loads_torsion, exp_torsion, rtol=1e-12, atol=1e-08)

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a combined displacement state
    (ua + ub) equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds.
    """
    E = 190000000000.0
    nu = 0.29
    A = 0.015
    I_y = 7.5e-06
    I_z = 6.4e-06
    J = 9.1e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (1.0, -0.5, 0.2)
    xj, yj, zj = (3.7, 1.4, 0.9)
    ua = np.array([0.01, -0.02, 0.03, 0.005, -0.004, 0.003, -0.01, 0.02, -0.03, -0.006, 0.0045, -0.0025])
    ub = np.array([-0.015, 0.01, 0.005, -0.002, 0.003, -0.004, 0.02, -0.01, 0.007, 0.003, -0.002, 0.001])
    fa = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua))
    fb = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ub))
    fab = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub))
    assert fa.shape == (12,)
    assert fb.shape == (12,)
    assert fab.shape == (12,)
    assert np.allclose(fab, fa + fb, rtol=1e-12, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance: If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """
    E = 205000000000.0
    nu = 0.28
    A = 0.02
    I_y = 6e-06
    I_z = 9e-06
    J = 7e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.2, -0.7, 0.4)
    xj, yj, zj = (1.3, 0.4, 1.0)
    u = np.array([0.01, 0.02, -0.005, 0.01, -0.02, 0.03, 0.06, 0.04, -0.015, -0.02, 0.035, -0.01])
    loads_ref = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u))
    alpha = -0.4
    gamma = 0.7
    Rx = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(alpha), -np.sin(alpha)], [0.0, np.sin(alpha), np.cos(alpha)]])
    Rz = np.array([[np.cos(gamma), -np.sin(gamma), 0.0], [np.sin(gamma), np.cos(gamma), 0.0], [0.0, 0.0, 1.0]])
    R = Rz @ Rx
    pi = np.array([xi, yi, zi])
    pj = np.array([xj, yj, zj])
    pi_r = R @ pi
    pj_r = R @ pj
    u_r = np.zeros(12)
    u_r[0:3] = R @ u[0:3]
    u_r[3:6] = R @ u[3:6]
    u_r[6:9] = R @ u[6:9]
    u_r[9:12] = R @ u[9:12]
    ele_info_r = dict(ele_info)
    ele_info_r['local_z'] = R @ ele_info['local_z']
    loads_rot = np.asarray(fcn(ele_info_r, pi_r[0], pi_r[1], pi_r[2], pj_r[0], pj_r[1], pj_r[2], u_r))
    assert loads_ref.shape == (12,)
    assert loads_rot.shape == (12,)
    assert np.allclose(loads_rot, loads_ref, rtol=1e-12, atol=1e-10)