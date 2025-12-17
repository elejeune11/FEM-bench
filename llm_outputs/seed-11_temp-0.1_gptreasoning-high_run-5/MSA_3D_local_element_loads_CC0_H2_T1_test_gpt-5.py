def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8e-06
    I_z = 5e-06
    J = 1.2e-05
    local_z = [0.0, 0.0, 1.0]
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (2.0, 0.0, 0.0)
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}
    t = np.array([0.123, -0.456, 0.789])
    u_dofs_global = np.zeros(12, dtype=float)
    u_dofs_global[0:3] = t
    u_dofs_global[6:9] = t
    loads_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert loads_local.shape == (12,)
    assert np.allclose(loads_local, np.zeros(12), rtol=1e-12, atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.02
    I_y = 3.4e-06
    I_z = 5.6e-06
    J = 7e-06
    local_z = [0.0, 0.0, 1.0]
    xi, yi, zi = (0.0, 0.0, 0.0)
    L = 2.0
    xj, yj, zj = (L, 0.0, 0.0)
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}
    u_axial = np.zeros(12, dtype=float)
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    f_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    k_ax = E * A / L
    assert np.isclose(f_axial[0], -k_ax, rtol=1e-12, atol=1e-09)
    assert np.isclose(f_axial[6], +k_ax, rtol=1e-12, atol=1e-09)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[0, 6]] = False
    assert np.allclose(f_axial[mask_other], 0.0, rtol=1e-12, atol=1e-09)
    u_shear_y = np.zeros(12, dtype=float)
    u_shear_y[1] = 0.0
    u_shear_y[7] = 1.0
    f_shear_y = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y)
    Fy_mag = 12.0 * E * I_z / L ** 3
    Mz_mag = 6.0 * E * I_z / L ** 2
    assert np.isclose(f_shear_y[1], -Fy_mag, rtol=1e-12, atol=1e-09)
    assert np.isclose(f_shear_y[5], -Mz_mag, rtol=1e-12, atol=1e-09)
    assert np.isclose(f_shear_y[7], +Fy_mag, rtol=1e-12, atol=1e-09)
    assert np.isclose(f_shear_y[11], +Mz_mag, rtol=1e-12, atol=1e-09)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[1, 5, 7, 11]] = False
    assert np.allclose(f_shear_y[mask_other], 0.0, rtol=1e-12, atol=1e-09)
    u_torsion = np.zeros(12, dtype=float)
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    f_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    G = E / (2.0 * (1.0 + nu))
    k_tor = G * J / L
    assert np.isclose(f_torsion[3], -k_tor, rtol=1e-12, atol=1e-09)
    assert np.isclose(f_torsion[9], +k_tor, rtol=1e-12, atol=1e-09)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[3, 9]] = False
    assert np.allclose(f_torsion[mask_other], 0.0, rtol=1e-12, atol=1e-09)

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a combined displacement state
    (ua + ub) equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds.
    """
    E = 200000000000.0
    nu = 0.28
    A = 0.015
    I_y = 4.5e-06
    I_z = 6e-06
    J = 8.5e-06
    local_z = [0.1, 0.3, 0.9]
    xi, yi, zi = (-1.2, 0.8, 0.5)
    xj, yj, zj = (1.3, -0.8, 1.3)
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}
    ua = np.array([0.1, -0.2, 0.05, 0.01, -0.02, 0.015, -0.05, 0.3, -0.1, -0.005, 0.01, -0.02], dtype=float)
    ub = np.array([-0.07, 0.12, -0.03, -0.02, 0.015, -0.01, 0.08, -0.05, 0.06, 0.012, -0.008, 0.017], dtype=float)
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fsum = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert fa.shape == (12,)
    assert fb.shape == (12,)
    assert fsum.shape == (12,)
    assert np.allclose(fsum, fa + fb, rtol=1e-10, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance: If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """

    def rodrigues_rotation_matrix(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
        I = np.eye(3)
        R = I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        return R
    E = 190000000000.0
    nu = 0.31
    A = 0.012
    I_y = 5.1e-06
    I_z = 4.4e-06
    J = 7.3e-06
    pi = np.array([0.1, -0.3, 0.2], dtype=float)
    pj = np.array([1.2, 0.5, 1.7], dtype=float)
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}
    u = np.array([0.08, -0.04, 0.02, 0.01, -0.015, 0.02, -0.06, 0.03, -0.05, -0.012, 0.006, -0.018], dtype=float)
    f_base = fcn(ele_info, pi[0], pi[1], pi[2], pj[0], pj[1], pj[2], u)
    axis = [0.35, 0.48, 0.81]
    angle = 0.7
    R = rodrigues_rotation_matrix(axis, angle)
    pi_r = R @ pi
    pj_r = R @ pj
    local_z_r = R @ local_z
    ui_trans = u[0:3]
    ui_rot = u[3:6]
    uj_trans = u[6:9]
    uj_rot = u[9:12]
    ui_trans_r = R @ ui_trans
    ui_rot_r = R @ ui_rot
    uj_trans_r = R @ uj_trans
    uj_rot_r = R @ uj_rot
    u_r = np.zeros(12, dtype=float)
    u_r[0:3] = ui_trans_r
    u_r[3:6] = ui_rot_r
    u_r[6:9] = uj_trans_r
    u_r[9:12] = uj_rot_r
    ele_info_r = dict(ele_info)
    ele_info_r['local_z'] = local_z_r
    f_rot = fcn(ele_info_r, pi_r[0], pi_r[1], pi_r[2], pj_r[0], pj_r[1], pj_r[2], u_r)
    assert f_base.shape == (12,)
    assert f_rot.shape == (12,)
    assert np.allclose(f_rot, f_base, rtol=1e-10, atol=1e-10)