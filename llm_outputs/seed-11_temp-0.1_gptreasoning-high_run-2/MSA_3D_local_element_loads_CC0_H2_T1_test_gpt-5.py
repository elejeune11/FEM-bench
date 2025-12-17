def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.02
    I_y = 4e-06
    I_z = 5e-06
    J = 7e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.0, 0.0, 0.0)
    L = 3.0
    xj, yj, zj = (L, 0.0, 0.0)
    tx, ty, tz = (0.123, -0.456, 0.789)
    u_dofs_global = np.array([tx, ty, tz, 0.0, 0.0, 0.0, tx, ty, tz, 0.0, 0.0, 0.0], dtype=float)
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-12, rtol=0.0)

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation
    """
    E = 200000000000.0
    nu = 0.25
    A = 0.002
    I_y = 8e-06
    I_z = 5e-06
    J = 7e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.0, 0.0, 0.0)
    L = 2.0
    xj, yj, zj = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[6] = 1.0
    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    EA_over_L = E * A / L
    expected_axial = np.zeros(12)
    expected_axial[0] = -EA_over_L
    expected_axial[6] = EA_over_L
    assert np.allclose(load_axial, expected_axial, rtol=1e-12, atol=1e-12)
    u_shear_y = np.zeros(12)
    u_shear_y[7] = 1.0
    load_shear_y = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y)
    EI_over_L3 = E * I_z / L ** 3
    EI_over_L2 = E * I_z / L ** 2
    expected_shear_y = np.zeros(12)
    expected_shear_y[1] = -12.0 * EI_over_L3
    expected_shear_y[5] = -6.0 * EI_over_L2
    expected_shear_y[7] = 12.0 * EI_over_L3
    expected_shear_y[11] = -6.0 * EI_over_L2
    assert np.allclose(load_shear_y, expected_shear_y, rtol=1e-12, atol=1e-12)
    u_torsion = np.zeros(12)
    u_torsion[9] = 1.0
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    G = E / (2.0 * (1.0 + nu))
    GJ_over_L = G * J / L
    expected_torsion = np.zeros(12)
    expected_torsion[3] = -GJ_over_L
    expected_torsion[9] = GJ_over_L
    assert np.allclose(load_torsion, expected_torsion, rtol=1e-12, atol=1e-12)

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a combined displacement state (ua + ub)
    equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds.
    """
    E = 210000000000.0
    nu = 0.29
    A = 0.015
    I_y = 6e-06
    I_z = 7e-06
    J = 9e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (1.7, 0.0, 0.0)
    ua = np.array([0.1, -0.2, 0.3, 0.01, 0.02, -0.03, -0.4, 0.5, -0.6, 0.04, -0.05, 0.06], dtype=float)
    ub = np.array([-0.3, 0.1, -0.2, -0.01, 0.03, 0.02, 0.7, -0.8, 0.9, -0.02, 0.01, -0.04], dtype=float)
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert np.allclose(fab, fa + fb, rtol=1e-12, atol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance: If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """

    def rodrigues_rotation_matrix(axis, angle):
        axis = axis / np.linalg.norm(axis)
        ax, ay, az = axis
        K = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], dtype=float)
        I = np.eye(3)
        return I * np.cos(angle) + np.sin(angle) * K + (1.0 - np.cos(angle)) * np.outer(axis, axis)

    def rotate_u(u, R):
        out = np.zeros_like(u)
        for n in range(2):
            t = u[6 * n:6 * n + 3]
            r = u[6 * n + 3:6 * n + 6]
            out[6 * n:6 * n + 3] = R @ t
            out[6 * n + 3:6 * n + 6] = R @ r
        return out
    E = 190000000000.0
    nu = 0.28
    A = 0.012
    I_y = 9e-06
    I_z = 6e-06
    J = 8e-06
    xi, yi, zi = (0.1, -0.2, 0.3)
    xj, yj, zj = (1.5, 0.8, -0.4)
    local_z0 = np.array([0.4, -0.9, 0.1], dtype=float)
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z0}
    u_global = np.array([0.2, -0.1, 0.05, 0.01, -0.02, 0.03, -0.15, 0.25, -0.35, -0.04, 0.05, -0.06], dtype=float)
    f_local_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_global)
    axis = np.array([0.3, -0.5, 0.8], dtype=float)
    angle = 0.7
    R = rodrigues_rotation_matrix(axis, angle)
    xi_rot, yi_rot, zi_rot = (R @ np.array([xi, yi, zi])).tolist()
    xj_rot, yj_rot, zj_rot = (R @ np.array([xj, yj, zj])).tolist()
    u_rot = rotate_u(u_global, R)
    local_z_rot = R @ local_z0
    ele_info_rot = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot}
    f_local_rotated = fcn(ele_info_rot, xi_rot, yi_rot, zi_rot, xj_rot, yj_rot, zj_rot, u_rot)
    assert np.allclose(f_local_original, f_local_rotated, rtol=1e-12, atol=1e-12)