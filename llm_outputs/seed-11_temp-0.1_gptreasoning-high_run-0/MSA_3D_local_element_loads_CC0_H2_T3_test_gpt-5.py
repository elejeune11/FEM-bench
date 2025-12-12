def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments."""
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.02, 'I_y': 2e-06, 'I_z': 3e-06, 'J': 4e-06}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    t = np.array([0.3, -0.5, 1.2])
    u_dofs_global = np.array([t[0], t[1], t[2], 0.0, 0.0, 0.0, t[0], t[1], t[2], 0.0, 0.0, 0.0])
    loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(loads, np.zeros(12), rtol=0, atol=1e-12)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    E = 210000000000.0
    nu = 0.3
    A = 0.02
    I_y = 2e-06
    I_z = 3e-06
    J = 4e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    L = xj - xi
    u_axial = np.zeros(12)
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    f_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    EA_over_L = E * A / L
    expected_axial = np.zeros(12)
    expected_axial[0] = -EA_over_L
    expected_axial[6] = EA_over_L
    assert np.allclose(f_axial, expected_axial, rtol=0, atol=1e-09)
    u_shear_y = np.zeros(12)
    u_shear_y[1] = 0.0
    u_shear_y[7] = 1.0
    f_shear_y = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y)
    EI_z = E * I_z
    Fy_mag = 12.0 * EI_z / L ** 3
    Mz_mag = 6.0 * EI_z / L ** 2
    expected_shear_y = np.zeros(12)
    expected_shear_y[1] = -Fy_mag
    expected_shear_y[5] = -Mz_mag
    expected_shear_y[7] = Fy_mag
    expected_shear_y[11] = Mz_mag
    assert np.allclose(f_shear_y, expected_shear_y, rtol=0, atol=1e-07)
    u_torsion = np.zeros(12)
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    f_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    G = E / (2.0 * (1.0 + nu))
    GJ_over_L = G * J / L
    expected_torsion = np.zeros(12)
    expected_torsion[3] = -GJ_over_L
    expected_torsion[9] = GJ_over_L
    assert np.allclose(f_torsion, expected_torsion, rtol=0, atol=1e-09)

def test_superposition_linearity(fcn):
    """Verify linearity: f(ua + ub) == f(ua) + f(ub) for arbitrary displacement states."""
    ele_info = {'E': 200000000000.0, 'nu': 0.29, 'A': 0.015, 'I_y': 1.7e-06, 'I_z': 2.1e-06, 'J': 3.5e-06}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.8, 0.0, 0.0)
    ua = np.array([0.1, -0.05, 0.02, 0.01, -0.02, 0.03, -0.04, 0.08, -0.03, 0.05, -0.04, 0.02])
    ub = np.array([-0.06, 0.03, -0.07, -0.02, 0.01, -0.01, 0.09, -0.02, 0.06, -0.03, 0.02, -0.05])
    f_ua = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    f_ub = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    f_sum = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert np.allclose(f_sum, f_ua + f_ub, rtol=1e-12, atol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 205000000000.0, 'nu': 0.28, 'A': 0.012, 'I_y': 1.9e-06, 'I_z': 2.3e-06, 'J': 3.1e-06, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.5, 0.0, 0.0)
    u = np.array([0.11, -0.07, 0.05, 0.02, -0.03, 0.04, -0.09, 0.12, -0.08, 0.06, 0.01, -0.05])

    def Rx(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]])

    def Ry(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, 0, sa], [0, 1, 0], [-sa, 0, ca]])

    def Rz(a):
        (ca, sa) = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0], [sa, ca, 0], [0, 0, 1]])
    R = Rz(0.57) @ Ry(-0.41) @ Rx(0.29)
    f_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u)
    (xi_rot, yi_rot, zi_rot) = (R @ np.array([xi, yi, zi])).tolist()
    (xj_rot, yj_rot, zj_rot) = (R @ np.array([xj, yj, zj])).tolist()
    u_rot = u.copy()
    u_rot[0:3] = R @ u[0:3]
    u_rot[3:6] = R @ u[3:6]
    u_rot[6:9] = R @ u[6:9]
    u_rot[9:12] = R @ u[9:12]
    ele_info_rot = dict(ele_info)
    ele_info_rot['local_z'] = R @ ele_info['local_z']
    f_local_rot = fcn(ele_info_rot, xi_rot, yi_rot, zi_rot, xj_rot, yj_rot, zj_rot, u_rot)
    assert np.allclose(f_local_rot, f_local, rtol=1e-10, atol=1e-10)