def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-06
    I_z = 1e-06
    J = 2e-06
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    tr = np.array([0.12, -0.34, 0.56])
    u_dofs_global = np.zeros(12)
    u_dofs_global[0:3] = tr
    u_dofs_global[6:9] = tr
    load = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load, np.zeros(12), atol=1e-12, rtol=0.0)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation"""
    E = 210000000000.0
    nu = 0.29
    G = E / (2.0 * (1.0 + nu))
    A = 0.02
    I_y = 2e-06
    I_z = 3e-06
    J = 4e-06
    L = 1.5
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    f_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    k_axial = E * A / L
    expected_axial = np.zeros(12)
    expected_axial[0] = -k_axial
    expected_axial[6] = +k_axial
    assert np.allclose(f_axial, expected_axial, atol=1e-09, rtol=0.0)
    u_shear_y = np.zeros(12)
    u_shear_y[1] = 0.0
    u_shear_y[7] = 1.0
    f_shear_y = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y)
    coef = E * I_z / L ** 3
    Fy1 = -12.0 * coef
    Mz1 = -6.0 * E * I_z / L ** 2
    Fy2 = +12.0 * coef
    Mz2 = -6.0 * E * I_z / L ** 2
    expected_shear_y = np.zeros(12)
    expected_shear_y[1] = Fy1
    expected_shear_y[5] = Mz1
    expected_shear_y[7] = Fy2
    expected_shear_y[11] = Mz2
    assert np.allclose(f_shear_y, expected_shear_y, atol=1e-09, rtol=0.0)
    u_torsion = np.zeros(12)
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    f_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    k_torsion = G * J / L
    expected_torsion = np.zeros(12)
    expected_torsion[3] = -k_torsion
    expected_torsion[9] = +k_torsion
    assert np.allclose(f_torsion, expected_torsion, atol=1e-09, rtol=0.0)

def test_superposition_linearity(fcn):
    """Verify linearity: f(ua + ub) equals f(ua) + f(ub) for a general 3D-oriented element."""
    E = 70000000000.0
    nu = 0.33
    A = 0.015
    I_y = 1.5e-06
    I_z = 2.5e-06
    J = 3e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 2.0, -1.0)
    ua = np.array([0.2, -0.1, 0.05, 0.01, -0.02, 0.03, -0.15, 0.25, -0.05, -0.02, 0.04, -0.01])
    ub = np.array([-0.3, 0.2, -0.1, 0.02, 0.01, -0.04, 0.35, -0.15, 0.2, -0.03, -0.02, 0.05])
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fsum_individual = fa + fb
    u_sum = ua + ub
    fsum_direct = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_sum)
    assert np.allclose(fsum_direct, fsum_individual, atol=1e-10, rtol=1e-12)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: rotating the entire configuration (coordinates, displacements, and local_z) by a rigid global rotation should not change the local internal end-load vector."""
    E = 200000000000.0
    nu = 0.31
    A = 0.012
    I_y = 1.8e-06
    I_z = 2.2e-06
    J = 2.6e-06
    (xi, yi, zi) = (1.0, 2.0, 3.0)
    (dx, dy, dz) = (0.7, -1.1, 0.5)
    (xj, yj, zj) = (xi + dx, yi + dy, zi + dz)
    local_z = np.array([0.2, 0.8, 0.5])
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}
    u_global = np.array([-0.12, 0.34, -0.56, 0.07, -0.03, 0.05, 0.22, -0.18, 0.41, -0.09, 0.06, -0.04])
    f_local_orig = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_global)
    axis = np.array([0.3, -0.7, 0.6])
    axis = axis / np.linalg.norm(axis)
    angle = np.deg2rad(40.0)
    K = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * (K @ K)
    xi_vec = np.array([xi, yi, zi])
    xj_vec = np.array([xj, yj, zj])
    xi_rot = R @ xi_vec
    xj_rot = R @ xj_vec
    local_z_rot = R @ local_z
    ele_info_rot = dict(ele_info)
    ele_info_rot['local_z'] = local_z_rot
    u_rot = np.zeros(12)
    u_rot[0:3] = R @ u_global[0:3]
    u_rot[3:6] = R @ u_global[3:6]
    u_rot[6:9] = R @ u_global[6:9]
    u_rot[9:12] = R @ u_global[9:12]
    f_local_rot = fcn(ele_info_rot, xi_rot[0], xi_rot[1], xi_rot[2], xj_rot[0], xj_rot[1], xj_rot[2], u_rot)
    assert np.allclose(f_local_orig, f_local_rot, atol=1e-09, rtol=1e-12)