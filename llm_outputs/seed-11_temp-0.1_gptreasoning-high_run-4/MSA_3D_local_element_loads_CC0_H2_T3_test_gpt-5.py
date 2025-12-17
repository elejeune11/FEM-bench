def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    E = 210000000000.0
    nu = 0.29
    A = 0.01
    I_y = 3e-06
    I_z = 2.5e-06
    J = 4e-06
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0.0, 0.0, 1.0]}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    t = np.array([0.123, -0.456, 0.789])
    u_dofs_global = np.hstack([t, np.zeros(3), t, np.zeros(3)])
    loads = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global), dtype=float)
    assert loads.shape == (12,)
    assert np.allclose(loads, 0.0, rtol=1e-12, atol=1e-12)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation"""
    E = 200000000000.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    A = 0.02
    I_y = 4e-06
    I_z = 3e-06
    J = 5e-06
    L = 2.5
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0.0, 0.0, 1.0]}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    loads_axial = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial), dtype=float)
    EA_over_L = E * A / L
    expected_axial = np.zeros(12)
    expected_axial[0] = -EA_over_L
    expected_axial[6] = +EA_over_L
    assert np.allclose(loads_axial, expected_axial, rtol=1e-12, atol=1e-12)
    u_shear_y = np.zeros(12)
    u_shear_y[1] = 0.0
    u_shear_y[7] = 1.0
    k = E * I_z
    Fy_i = -12.0 * k / L ** 3
    Mz_i = -6.0 * k / L ** 2
    Fy_j = +12.0 * k / L ** 3
    Mz_j = -6.0 * k / L ** 2
    expected_shear_y = np.zeros(12)
    expected_shear_y[1] = Fy_i
    expected_shear_y[5] = Mz_i
    expected_shear_y[7] = Fy_j
    expected_shear_y[11] = Mz_j
    loads_shear_y = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y), dtype=float)
    assert np.allclose(loads_shear_y, expected_shear_y, rtol=1e-12, atol=1e-12)
    u_torsion = np.zeros(12)
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    loads_torsion = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion), dtype=float)
    GJ_over_L = G * J / L
    expected_torsion = np.zeros(12)
    expected_torsion[3] = -GJ_over_L
    expected_torsion[9] = +GJ_over_L
    assert np.allclose(loads_torsion, expected_torsion, rtol=1e-12, atol=1e-12)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a combined displacement state (ua + ub)
    equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds."""
    E = 190000000000.0
    nu = 0.28
    A = 0.015
    I_y = 3.3e-06
    I_z = 2.7e-06
    J = 4.8e-06
    L = 3.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0.0, 0.0, 1.0]}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    ua = np.array([0.1, 0.2, -0.1, 0.05, 0.02, -0.03, -0.04, -0.1, 0.05, 0.02, 0.03, 0.01], dtype=float)
    ub = np.array([-0.03, 0.07, 0.04, -0.02, 0.01, 0.06, 0.08, -0.02, -0.03, 0.04, -0.05, 0.02], dtype=float)
    uab = ua + ub
    fa = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua), dtype=float)
    fb = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ub), dtype=float)
    fab = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, uab), dtype=float)
    assert fa.shape == (12,) and fb.shape == (12,) and (fab.shape == (12,))
    assert np.allclose(fab, fa + fb, rtol=1e-12, atol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: Rotating the entire configuration (coords, displacements, and local_z) by a rigid global rotation
    should leave the local internal end-load vector unchanged."""
    E = 205000000000.0
    nu = 0.27
    A = 0.012
    I_y = 3.8e-06
    I_z = 2.2e-06
    J = 4.1e-06
    L = 2.2
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z.tolist()}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    u_global = np.array([0.15, -0.12, 0.05, 0.03, -0.04, 0.06, -0.08, 0.09, -0.02, -0.05, 0.07, -0.03], dtype=float)

    def rotation_matrix(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
        I = np.eye(3)
        return I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    axis = [1.0, 0.3, -0.2]
    angle = 0.6
    R = rotation_matrix(axis, angle)
    pi = np.array([xi, yi, zi])
    pj = np.array([xj, yj, zj])
    pi_rot = R @ pi
    pj_rot = R @ pj
    local_z_rot = R @ local_z

    def rotate_dofs(u):
        u = np.asarray(u, dtype=float)
        t1 = R @ u[0:3]
        r1 = R @ u[3:6]
        t2 = R @ u[6:9]
        r2 = R @ u[9:12]
        return np.hstack([t1, r1, t2, r2])
    u_rot = rotate_dofs(u_global)
    f_local = np.asarray(fcn(ele_info, *pi, *pj, u_global), dtype=float)
    ele_info_rot = dict(ele_info)
    ele_info_rot['local_z'] = local_z_rot.tolist()
    f_local_rot = np.asarray(fcn(ele_info_rot, *pi_rot, *pj_rot, u_rot), dtype=float)
    assert f_local.shape == (12,) and f_local_rot.shape == (12,)
    assert np.allclose(f_local, f_local_rot, rtol=1e-11, atol=1e-11)