def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 2e-06
    I_z = 3e-06
    J = 5e-06
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    t = np.array([0.7, -0.3, 2.2])
    u_dofs_global = np.zeros(12)
    u_dofs_global[0:3] = t
    u_dofs_global[6:9] = t
    loads_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(loads_local, np.zeros(12), atol=1e-12, rtol=0.0)

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 2e-06
    I_z = 3e-06
    J = 5e-06
    G = E / (2.0 * (1.0 + nu))
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    f_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    k_axial = E * A / L
    assert np.isclose(abs(f_axial[0]), k_axial, rtol=0, atol=1e-10)
    assert np.isclose(abs(f_axial[6]), k_axial, rtol=0, atol=1e-10)
    assert np.isclose(f_axial[0] + f_axial[6], 0.0, atol=1e-12)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[0, 6]] = False
    assert np.allclose(f_axial[mask_other], 0.0, atol=1e-10)
    u_shear = np.zeros(12)
    u_shear[1] = 0.0
    u_shear[7] = 1.0
    f_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    Fy_mag = 12.0 * E * I_z / L ** 3
    Mz_mag = 6.0 * E * I_z / L ** 2
    assert np.isclose(abs(f_shear[1]), Fy_mag, atol=1e-08)
    assert np.isclose(abs(f_shear[7]), Fy_mag, atol=1e-08)
    assert np.isclose(f_shear[1] + f_shear[7], 0.0, atol=1e-08)
    assert np.isclose(abs(f_shear[5]), Mz_mag, atol=1e-08)
    assert np.isclose(abs(f_shear[11]), Mz_mag, atol=1e-08)
    assert np.isclose(f_shear[5] - f_shear[11], 0.0, atol=1e-08)
    mask_zero = np.ones(12, dtype=bool)
    mask_zero[[1, 5, 7, 11]] = False
    assert np.allclose(f_shear[mask_zero], 0.0, atol=1e-08)
    u_torsion = np.zeros(12)
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    f_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    k_torsion = G * J / L
    assert np.isclose(abs(f_torsion[3]), k_torsion, atol=1e-10)
    assert np.isclose(abs(f_torsion[9]), k_torsion, atol=1e-10)
    assert np.isclose(f_torsion[3] + f_torsion[9], 0.0, atol=1e-12)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[3, 9]] = False
    assert np.allclose(f_torsion[mask_other], 0.0, atol=1e-10)

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a combined displacement state (ua + ub)
    equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 2e-06
    I_z = 3e-06
    J = 5e-06
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    ua = np.array([0.1, 0.0, 0.2, 0.0, 0.0, 0.05, 0.4, -0.1, 0.0, 0.1, 0.0, -0.02], dtype=float)
    ub = np.array([-0.3, 0.2, -0.1, 0.03, -0.01, 0.0, 0.0, 0.25, 0.15, -0.05, 0.02, 0.04], dtype=float)
    uab = ua + ub
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, uab)
    assert np.allclose(fab, fa + fb, atol=1e-10, rtol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance: If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 2e-06
    I_z = 3e-06
    J = 5e-06
    L = 2.0
    local_z = np.array([0.0, 0.0, 1.0])
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    u = np.array([0.2, -0.3, 0.1, 0.05, -0.04, 0.02, -0.1, 0.4, 0.3, 0.01, 0.02, -0.03], dtype=float)
    f_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u)
    axis = np.array([1.0, 2.0, 3.0], dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = 0.7
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + np.sin(theta) * K + (1.0 - np.cos(theta)) * (K @ K)
    (xi_r, yi_r, zi_r) = (R @ np.array([xi, yi, zi])).tolist()
    (xj_r, yj_r, zj_r) = (R @ np.array([xj, yj, zj])).tolist()
    local_z_r = R @ local_z
    u_r = np.zeros(12)
    for n in range(2):
        t = u[6 * n:6 * n + 3]
        r = u[6 * n + 3:6 * n + 6]
        u_r[6 * n:6 * n + 3] = R @ t
        u_r[6 * n + 3:6 * n + 6] = R @ r
    ele_info_r = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_r}
    f_local_rot = fcn(ele_info_r, xi_r, yi_r, zi_r, xj_r, yj_r, zj_r, u_r)
    assert np.allclose(f_local_rot, f_local, atol=1e-09, rtol=1e-12)