def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    E = 2.0
    nu = 0.3
    A = 1.0
    I_y = 0.5
    I_z = 1.0
    J = 0.25
    ele_info = dict(E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=[0.0, 0.0, 1.0])
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    trans = np.array([0.53, -1.12, 2.3], dtype=float)
    u = np.zeros(12, dtype=float)
    u[0:3] = trans
    u[6:9] = trans
    f_local = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u))
    assert f_local.shape == (12,)
    assert np.allclose(f_local, np.zeros(12), atol=1e-08, rtol=1e-08)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation
    """
    E = 2.0
    nu = 0.3
    A = 1.0
    I_y = 0.5
    I_z = 1.0
    J = 0.25
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    ele_info = dict(E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=[0.0, 0.0, 1.0])
    L = np.linalg.norm(np.array([xj - xi, yj - yi, zj - zi]))
    ua = np.zeros(12, dtype=float)
    ua[0] = 0.0
    ua[6] = 1.0
    f_axial = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua))
    expected_axial = np.zeros(12)
    expected_axial[0] = -E * A / L
    expected_axial[6] = E * A / L
    assert np.allclose(f_axial, expected_axial, atol=1e-10, rtol=1e-12)
    ub = np.zeros(12, dtype=float)
    ub[1] = 0.0
    ub[7] = 1.0
    f_shear = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ub))
    Kb = E * I_z / L ** 3 * np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]], dtype=float)
    u4 = np.array([ub[1], ub[5], ub[7], ub[11]], dtype=float)
    f4 = Kb.dot(u4)
    expected_shear = np.zeros(12)
    expected_shear[[1, 5, 7, 11]] = f4
    assert np.allclose(f_shear, expected_shear, atol=1e-10, rtol=1e-12)
    uc = np.zeros(12, dtype=float)
    uc[3] = 0.0
    uc[9] = 1.0
    f_torsion = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, uc))
    G = E / (2.0 * (1.0 + nu))
    Kt = G * J / L
    expected_torsion = np.zeros(12)
    expected_torsion[3] = -Kt
    expected_torsion[9] = Kt
    assert np.allclose(f_torsion, expected_torsion, atol=1e-10, rtol=1e-12)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a combined displacement state (ua + ub) equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds."""
    E = 2.0
    nu = 0.3
    A = 1.0
    I_y = 0.5
    I_z = 1.0
    J = 0.25
    ele_info = dict(E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=[0.0, 0.0, 1.0])
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    ua = np.zeros(12, dtype=float)
    ua[0] = 0.2
    ua[6] = 0.35
    ua[1] = -0.05
    ua[7] = 0.12
    ua[3] = 0.015
    ua[9] = -0.01
    ub = np.zeros(12, dtype=float)
    ub[2] = -0.08
    ub[8] = 0.1
    ub[5] = 0.02
    ub[11] = -0.015
    f_a = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua))
    f_b = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ub))
    f_ab = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub))
    assert np.allclose(f_ab, f_a + f_b, atol=1e-10, rtol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z) by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """
    E = 2.0
    nu = 0.3
    A = 1.0
    I_y = 0.5
    I_z = 1.0
    J = 0.25
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.3, -0.1)
    local_z = np.array([0.1, 0.2, 1.0], dtype=float)
    ele_info = dict(E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=local_z.tolist())
    u = np.zeros(12, dtype=float)
    u[0:3] = np.array([0.12, -0.07, 0.05])
    u[6:9] = np.array([-0.04, 0.09, -0.02])
    u[3:6] = np.array([0.01, -0.02, 0.03])
    u[9:12] = np.array([-0.02, 0.015, -0.01])
    f_orig = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u))
    axis = np.array([1.0, 2.0, 3.0], dtype=float)
    angle = 0.73
    R = _rot_matrix_from_axis_angle(axis, angle)
    ri = R.dot(np.array([xi, yi, zi], dtype=float))
    rj = R.dot(np.array([xj, yj, zj], dtype=float))
    rz = R.dot(local_z)
    ur = np.zeros(12, dtype=float)
    ur[0:3] = R.dot(u[0:3])
    ur[6:9] = R.dot(u[6:9])
    ur[3:6] = R.dot(u[3:6])
    ur[9:12] = R.dot(u[9:12])
    ele_info_r = dict(E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=rz.tolist())
    f_rot = np.asarray(fcn(ele_info_r, float(ri[0]), float(ri[1]), float(ri[2]), float(rj[0]), float(rj[1]), float(rj[2]), ur))
    assert f_orig.shape == (12,)
    assert f_rot.shape == (12,)
    assert np.allclose(f_orig, f_rot, atol=1e-08, rtol=1e-08)