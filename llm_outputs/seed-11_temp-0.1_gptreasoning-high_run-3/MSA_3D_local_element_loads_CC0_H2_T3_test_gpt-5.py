def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 4e-06
    Iz = 5e-06
    J = 1.2e-05
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (2.0, 0.0, 0.0)
    ux, uy, uz = (0.5, -1.2, 2.3)
    u_dofs_global = np.array([ux, uy, uz, 0.0, 0.0, 0.0, ux, uy, uz, 0.0, 0.0, 0.0], dtype=float)
    loads_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    loads_local = np.asarray(loads_local, dtype=float)
    assert loads_local.shape == (12,)
    assert np.allclose(loads_local, np.zeros(12), atol=1e-12, rtol=0.0)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation"""
    E = 210000000000.0
    nu = 0.3
    A = 0.003
    Iy = 4e-06
    Iz = 5e-06
    J = 1.2e-05
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    f_axial = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial), dtype=float)
    Nx = E * A / L
    assert np.isclose(f_axial[0], Nx * (u_axial[0] - u_axial[6]), rtol=1e-10, atol=1e-10 * abs(Nx))
    assert np.isclose(f_axial[6], Nx * (u_axial[6] - u_axial[0]), rtol=1e-10, atol=1e-10 * abs(Nx))
    mask_other = np.ones(12, dtype=bool)
    mask_other[[0, 6]] = False
    assert np.allclose(f_axial[mask_other], 0.0, atol=1e-09, rtol=0.0)
    u_shear_y = np.zeros(12)
    u_shear_y[1] = 0.0
    u_shear_y[7] = 1.0
    f_shear_y = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y), dtype=float)
    EIz = E * Iz
    Fy_mag = 12.0 * EIz / L ** 3
    Mz_mag = 6.0 * EIz / L ** 2
    assert np.isclose(abs(f_shear_y[1]), Fy_mag, rtol=1e-10, atol=1e-08)
    assert np.isclose(f_shear_y[1] + f_shear_y[7], 0.0, atol=1e-08, rtol=0.0)
    assert np.isclose(f_shear_y[5], f_shear_y[11], atol=1e-08, rtol=0.0)
    assert np.isclose(abs(f_shear_y[5]), Mz_mag, rtol=1e-10, atol=1e-08)
    assert np.isclose(f_shear_y[5], 0.5 * L * f_shear_y[1], atol=1e-08, rtol=1e-10)
    mask_other2 = np.ones(12, dtype=bool)
    mask_other2[[1, 5, 7, 11]] = False
    assert np.allclose(f_shear_y[mask_other2], 0.0, atol=1e-08, rtol=0.0)
    u_torsion = np.zeros(12)
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    f_torsion = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion), dtype=float)
    G = E / (2.0 * (1.0 + nu))
    T = G * J / L
    assert np.isclose(abs(f_torsion[3]), abs(T), rtol=1e-10, atol=1e-08)
    assert np.isclose(f_torsion[3] + f_torsion[9], 0.0, atol=1e-08, rtol=0.0)
    mask_other3 = np.ones(12, dtype=bool)
    mask_other3[[3, 9]] = False
    assert np.allclose(f_torsion[mask_other3], 0.0, atol=1e-08, rtol=0.0)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a combined displacement state (ua + ub)
    equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds."""
    E = 200000000000.0
    nu = 0.29
    A = 0.0022
    Iy = 3.5e-06
    Iz = 4e-06
    J = 1.1e-05
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (1.0, 2.0, -0.5)
    xj, yj, zj = (3.0, -1.0, 2.0)
    ua = np.array([0.1, -0.2, 0.3, 0.05, -0.02, 0.04, -0.1, 0.08, -0.06, 0.03, -0.04, 0.02], dtype=float)
    ub = np.array([-0.05, 0.1, 0.0, -0.02, 0.03, -0.01, 0.09, -0.07, 0.05, -0.02, 0.06, -0.03], dtype=float)
    uc = ua + ub
    fa = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua), dtype=float)
    fb = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ub), dtype=float)
    fc = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, uc), dtype=float)
    assert fa.shape == (12,)
    assert fb.shape == (12,)
    assert fc.shape == (12,)
    assert np.allclose(fc, fa + fb, rtol=1e-10, atol=1e-09)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z) by a rigid global rotation R,
    the local internal end-load vector should be unchanged."""
    E = 210000000000.0
    nu = 0.3
    A = 0.0025
    Iy = 3.2e-06
    Iz = 4.1e-06
    J = 1.3e-05
    local_z = np.array([0.2, -0.1, 0.97], dtype=float)
    local_z = local_z / np.linalg.norm(local_z)
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z.copy()}
    xi, yi, zi = (-1.2, 0.7, 2.3)
    xj, yj, zj = (1.5, 2.1, -0.4)
    u_global = np.array([0.12, -0.23, 0.34, 0.05, -0.04, 0.03, -0.11, 0.09, -0.07, 0.02, 0.06, -0.05], dtype=float)
    f_local = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_global), dtype=float)

    def rot_matrix(axis, angle):
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        ax, ay, az = a
        K = np.array([[0, -az, ay], [az, 0, -ax], [-ay, ax, 0]], dtype=float)
        I = np.eye(3)
        c = np.cos(angle)
        s = np.sin(angle)
        return c * I + s * K + (1.0 - c) * np.outer(a, a)
    R = rot_matrix(axis=[-0.3, 0.7, 0.64], angle=0.83)
    pi = np.array([xi, yi, zi], dtype=float)
    pj = np.array([xj, yj, zj], dtype=float)
    pi_rot = R @ pi
    pj_rot = R @ pj
    u1 = u_global[:3]
    th1 = u_global[3:6]
    u2 = u_global[6:9]
    th2 = u_global[9:12]
    u1r = R @ u1
    th1r = R @ th1
    u2r = R @ u2
    th2r = R @ th2
    u_rot = np.concatenate([u1r, th1r, u2r, th2r])
    local_z_rot = R @ local_z
    ele_info_rot = {'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z_rot}
    f_local_rot = np.asarray(fcn(ele_info_rot, pi_rot[0], pi_rot[1], pi_rot[2], pj_rot[0], pj_rot[1], pj_rot[2], u_rot), dtype=float)
    assert f_local.shape == (12,)
    assert f_local_rot.shape == (12,)
    assert np.allclose(f_local, f_local_rot, rtol=1e-10, atol=1e-09)