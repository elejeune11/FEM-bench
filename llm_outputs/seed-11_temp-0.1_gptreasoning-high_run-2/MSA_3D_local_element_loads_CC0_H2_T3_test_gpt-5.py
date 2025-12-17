def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element produces zero internal
    forces and moments in the local load vector.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 6e-06
    I_z = 8e-06
    J = 1.2e-05
    ele_info = dict(E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J)
    L = 2.0
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    t = np.array([0.23, -1.1, 0.7])
    u_dofs_global = np.array([t[0], t[1], t[2], 0.0, 0.0, 0.0, t[0], t[1], t[2], 0.0, 0.0, 0.0], dtype=float)
    loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert isinstance(loads, np.ndarray)
    assert loads.shape == (12,)
    assert np.allclose(loads, 0.0, atol=1e-12)

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation
    Verifies expected magnitudes and sign relationships of internal end loads in local coordinates.
    """
    E = 200000000000.0
    nu = 0.3
    G = E / (2.0 * (1.0 + nu))
    A = 0.01
    I_y = 7e-06
    I_z = 5e-06
    J = 1.1e-05
    ele_info = dict(E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J)
    L = 2.0
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    loads_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    k_axial = E * A / L
    Fx_i, Fx_j = (loads_axial[0], loads_axial[6])
    assert np.allclose(abs(Fx_i), k_axial, rtol=0, atol=1e-09 * k_axial)
    assert np.allclose(abs(Fx_j), k_axial, rtol=0, atol=1e-09 * k_axial)
    assert np.allclose(Fx_i + Fx_j, 0.0, atol=1e-10 * k_axial)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[0, 6]] = False
    assert np.allclose(loads_axial[mask_other], 0.0, atol=1e-10 * k_axial)
    u_shear = np.zeros(12)
    u_shear[1] = 0.0
    u_shear[7] = 1.0
    loads_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    kV = 12.0 * E * I_z / L ** 3
    kM = 6.0 * E * I_z / L ** 2
    Fy_i, Fy_j = (loads_shear[1], loads_shear[7])
    Mz_i, Mz_j = (loads_shear[5], loads_shear[11])
    assert np.allclose(Fy_i + Fy_j, 0.0, atol=1e-10 * max(1.0, kV))
    assert np.allclose([abs(Fy_i), abs(Fy_j)], [kV, kV], rtol=0, atol=1e-09 * kV)
    assert np.allclose(Mz_i - Mz_j, 0.0, atol=1e-10 * max(1.0, kM))
    assert np.allclose([abs(Mz_i), abs(Mz_j)], [kM, kM], rtol=0, atol=1e-09 * kM)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[1, 7, 5, 11]] = False
    atol_other = 1e-10 * max(kV, kM, 1.0)
    assert np.allclose(loads_shear[mask_other], 0.0, atol=atol_other)
    u_torsion = np.zeros(12)
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    loads_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    kT = G * J / L
    Mx_i, Mx_j = (loads_torsion[3], loads_torsion[9])
    assert np.allclose(abs(Mx_i), kT, rtol=0, atol=1e-09 * kT)
    assert np.allclose(abs(Mx_j), kT, rtol=0, atol=1e-09 * kT)
    assert np.allclose(Mx_i + Mx_j, 0.0, atol=1e-10 * kT)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[3, 9]] = False
    assert np.allclose(loads_torsion[mask_other], 0.0, atol=1e-10 * kT)

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: internal loads for a combined displacement state
    ua + ub equal f(ua) + f(ub), confirming superposition holds.
    """
    E = 210000000000.0
    nu = 0.29
    A = 0.012
    I_y = 5e-06
    I_z = 4e-06
    J = 9e-06
    ele_info = dict(E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J)
    L = 3.5
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    rng = np.random.default_rng(42)
    ua = rng.standard_normal(12)
    ub = rng.standard_normal(12)
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert isinstance(fa, np.ndarray) and fa.shape == (12,)
    assert isinstance(fb, np.ndarray) and fb.shape == (12,)
    assert isinstance(fab, np.ndarray) and fab.shape == (12,)
    assert np.allclose(fab, fa + fb, atol=1e-09 * max(1.0, np.linalg.norm(fa) + np.linalg.norm(fb)))

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance:
    If we rotate the entire configuration (node coordinates, displacements, and local_z)
    by a rigid rotation R, the local internal end-load vector should be unchanged.
    """

    def rotmat_from_axis_angle(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        c = np.cos(angle)
        s = np.sin(angle)
        C = 1.0 - c
        return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s], [z * x * C - y * s, z * y * C + x * s, c + z * z * C]], dtype=float)
    E = 190000000000.0
    nu = 0.28
    A = 0.009
    I_y = 3.5e-06
    I_z = 4.2e-06
    J = 8e-06
    xi, yi, zi = (3.0, -1.0, 0.5)
    dir_vec = np.array([2.0, 0.3, 1.1], dtype=float)
    xj, yj, zj = (xi + dir_vec[0], yi + dir_vec[1], zi + dir_vec[2])
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)
    rng = np.random.default_rng(7)
    u = rng.standard_normal(12)
    ele_info = dict(E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=local_z)
    loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, u)
    axis = np.array([1.0, 1.0, 1.0], dtype=float)
    angle = 0.7
    R = rotmat_from_axis_angle(axis, angle)
    pi = np.array([xi, yi, zi], dtype=float)
    pj = np.array([xj, yj, zj], dtype=float)
    pi_r = R @ pi
    pj_r = R @ pj
    local_z_r = R @ local_z
    u_r = np.zeros(12, dtype=float)
    u_r[0:3] = R @ u[0:3]
    u_r[3:6] = R @ u[3:6]
    u_r[6:9] = R @ u[6:9]
    u_r[9:12] = R @ u[9:12]
    ele_info_r = dict(E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=local_z_r)
    loads_r = fcn(ele_info_r, pi_r[0], pi_r[1], pi_r[2], pj_r[0], pj_r[1], pj_r[2], u_r)
    assert isinstance(loads_r, np.ndarray) and loads_r.shape == (12,)
    assert np.allclose(loads, loads_r, atol=1e-09 * max(1.0, np.linalg.norm(loads)))