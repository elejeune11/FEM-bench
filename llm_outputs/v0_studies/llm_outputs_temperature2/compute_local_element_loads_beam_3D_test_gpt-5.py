def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments
    in the local load vector.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-06
    I_z = 1e-06
    J = 2e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    L = 2.0
    (xj, yj, zj) = (L, 0.0, 0.0)
    T = np.array([0.2, -0.35, 0.5])
    u1 = np.array([T[0], T[1], T[2], 0.0, 0.0, 0.0])
    u2 = np.array([T[0], T[1], T[2], 0.0, 0.0, 0.0])
    u_dofs_global = np.hstack([u1, u2])
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
    E = 200000000000.0
    nu = 0.29
    A = 0.02
    I_y = 2e-06
    I_z = 3e-06
    J = 5e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    L = 3.0
    (xj, yj, zj) = (L, 0.0, 0.0)
    u_ax = np.zeros(12)
    u_ax[0] = 0.0
    u_ax[6] = 1.0
    f_ax = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_ax)
    EA_over_L = E * A / L
    f_ax_expected = np.zeros(12)
    f_ax_expected[0] = -EA_over_L
    f_ax_expected[6] = +EA_over_L
    assert np.allclose(f_ax, f_ax_expected, rtol=1e-12, atol=1e-12)
    u_shear_y = np.zeros(12)
    u_shear_y[1] = 0.0
    u_shear_y[7] = 1.0
    f_shear_y = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y)
    k = E * I_z / L ** 3
    f_shear_y_expected = np.zeros(12)
    f_shear_y_expected[1] = -12.0 * k
    f_shear_y_expected[5] = -6.0 * L * k
    f_shear_y_expected[7] = +12.0 * k
    f_shear_y_expected[11] = -6.0 * L * k
    assert np.allclose(f_shear_y, f_shear_y_expected, rtol=1e-12, atol=1e-12)
    u_tors = np.zeros(12)
    u_tors[3] = 0.0
    u_tors[9] = 1.0
    f_tors = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_tors)
    G = E / (2.0 * (1.0 + nu))
    GJ_over_L = G * J / L
    f_tors_expected = np.zeros(12)
    f_tors_expected[3] = -GJ_over_L
    f_tors_expected[9] = +GJ_over_L
    assert np.allclose(f_tors, f_tors_expected, rtol=1e-12, atol=1e-12)

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a combined displacement state
    (ua + ub) equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds.
    """
    E = 70000000000.0
    nu = 0.33
    A = 0.003
    I_y = 8e-07
    I_z = 1.1e-06
    J = 9e-07
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    L = 1.7
    (xj, yj, zj) = (L, 0.0, 0.0)
    ua = np.array([0.01, -0.02, 0.03, 0.04, -0.01, 0.02, -0.03, 0.01, -0.02, 0.05, -0.02, 0.01])
    ub = np.array([-0.02, 0.01, -0.01, -0.03, 0.02, -0.04, 0.04, -0.03, 0.02, -0.01, 0.03, -0.02])
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert np.allclose(fab, fa + fb, rtol=1e-12, atol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance: If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """

    def rot_x(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])

    def rot_y(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, 0.0, sa], [0.0, 1.0, 0.0], [-sa, 0.0, ca]])

    def rot_z(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    E = 100000000000.0
    nu = 0.28
    A = 0.015
    I_y = 2.5e-06
    I_z = 1.7e-06
    J = 3.2e-06
    local_z0 = np.array([0.0, 0.0, 1.0])
    ele_info0 = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z0}
    (xi0, yi0, zi0) = (0.0, 0.0, 0.0)
    L = 2.3
    (xj0, yj0, zj0) = (L, 0.0, 0.0)
    u0 = np.array([0.1, -0.2, 0.3, 0.05, -0.07, 0.02, -0.4, 0.6, -0.1, -0.03, 0.08, -0.09])
    f0 = fcn(ele_info0, xi0, yi0, zi0, xj0, yj0, zj0, u0)
    (ax, ay, az) = (0.3, -0.2, 0.4)
    R = rot_z(az) @ rot_y(ay) @ rot_x(ax)
    (xi_rot, yi_rot, zi_rot) = (R @ np.array([xi0, yi0, zi0])).tolist()
    (xj_rot, yj_rot, zj_rot) = (R @ np.array([xj0, yj0, zj0])).tolist()
    local_z_rot = R @ local_z0
    ele_info_rot = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot}
    u_rot = np.zeros(12)
    u_rot[0:3] = R @ u0[0:3]
    u_rot[3:6] = R @ u0[3:6]
    u_rot[6:9] = R @ u0[6:9]
    u_rot[9:12] = R @ u0[9:12]
    f_rot = fcn(ele_info_rot, xi_rot, yi_rot, zi_rot, xj_rot, yj_rot, zj_rot, u_rot)
    assert np.allclose(f_rot, f0, rtol=1e-12, atol=1e-12)