def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments."""
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 2e-06
    I_z = 3e-06
    J = 1.5e-06
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    t = np.array([0.5, -1.2, 3.4])
    u_dofs_global = np.zeros(12)
    u_dofs_global[0:3] = t
    u_dofs_global[6:9] = t
    loads_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(loads_local, 0.0, atol=1e-12, rtol=1e-12)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via v2 - v1 with zero rotations
    (3) Unit torsional rotation"""
    E = 200000000000.0
    nu = 0.3
    G = E / (2 * (1 + nu))
    A = 0.02
    I_y = 4e-06
    I_z = 5e-06
    J = 7e-06
    L = 3.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    loads_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    expected_axial = np.zeros(12)
    EA_over_L = E * A / L
    expected_axial[0] = -EA_over_L
    expected_axial[6] = +EA_over_L
    assert np.allclose(loads_axial, expected_axial, rtol=1e-09, atol=1e-12)
    u_shear_y = np.zeros(12)
    u_shear_y[1] = 0.0
    u_shear_y[7] = 1.0
    loads_shear_y = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y)
    expected_shear_y = np.zeros(12)
    EIz = E * I_z
    L2 = L * L
    L3 = L2 * L
    expected_shear_y[1] = -12.0 * EIz / L3
    expected_shear_y[5] = -6.0 * EIz / L2
    expected_shear_y[7] = +12.0 * EIz / L3
    expected_shear_y[11] = -6.0 * EIz / L2
    assert np.allclose(loads_shear_y, expected_shear_y, rtol=1e-09, atol=1e-12)
    u_torsion = np.zeros(12)
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    loads_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    expected_torsion = np.zeros(12)
    GJ_over_L = G * J / L
    expected_torsion[3] = -GJ_over_L
    expected_torsion[9] = +GJ_over_L
    assert np.allclose(loads_torsion, expected_torsion, rtol=1e-09, atol=1e-12)

def test_superposition_linearity(fcn):
    """Verify linearity: f(ua + ub) == f(ua) + f(ub)."""
    E = 210000000000.0
    nu = 0.29
    A = 0.015
    I_y = 3.5e-06
    I_z = 2.8e-06
    J = 4.2e-06
    L = 2.5
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    rng = np.random.default_rng(12345)
    ua = rng.normal(scale=0.1, size=12)
    ub = rng.normal(scale=0.1, size=12)
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert np.allclose(fab, fa + fb, rtol=1e-10, atol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance under global rigid rotation: rotating coordinates, displacements, and local_z leaves the local end-load vector unchanged."""
    E = 205000000000.0
    nu = 0.31
    A = 0.012
    I_y = 3e-06
    I_z = 6e-06
    J = 5e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}
    (xi, yi, zi) = (0.2, -0.4, 1.3)
    L = 1.7
    dx = 1.0
    dy = 0.2
    dz = -0.3
    norm = np.linalg.norm([dx, dy, dz])
    (dx, dy, dz) = L * np.array([dx, dy, dz]) / norm
    (xj, yj, zj) = (xi + dx, yi + dy, zi + dz)
    local_z = np.array([0.0, 0.0, 1.0])
    ele_info['local_z'] = local_z
    u = np.array([0.1, -0.2, 0.05, 0.02, -0.01, 0.03, -0.04, 0.21, -0.11, -0.02, 0.06, -0.03])
    f_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u)

    def rotmat_xyz(ax, ay, az):
        (cx, sx) = (np.cos(ax), np.sin(ax))
        (cy, sy) = (np.cos(ay), np.sin(ay))
        (cz, sz) = (np.cos(az), np.sin(az))
        Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
        Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
        Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])
        return Rz @ Ry @ Rx
    R = rotmat_xyz(0.3, -0.4, 0.2)
    ri = R @ np.array([xi, yi, zi])
    rj = R @ np.array([xj, yj, zj])
    (xi_r, yi_r, zi_r) = ri.tolist()
    (xj_r, yj_r, zj_r) = rj.tolist()
    local_z_r = R @ local_z
    ele_info_r = ele_info.copy()
    ele_info_r['local_z'] = local_z_r
    u_r = np.zeros(12)
    u_r[0:3] = R @ u[0:3]
    u_r[3:6] = R @ u[3:6]
    u_r[6:9] = R @ u[6:9]
    u_r[9:12] = R @ u[9:12]
    f_local_r = fcn(ele_info_r, xi_r, yi_r, zi_r, xj_r, yj_r, zj_r, u_r)
    assert np.allclose(f_local, f_local_r, rtol=1e-10, atol=1e-12)