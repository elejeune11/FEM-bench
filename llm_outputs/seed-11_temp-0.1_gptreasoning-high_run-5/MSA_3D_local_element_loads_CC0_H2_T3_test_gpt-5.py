def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    import numpy as np
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8e-06
    I_z = 4e-06
    J = 1.5e-05
    L = 2.5
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    t = np.array([0.11, -0.24, 0.35], dtype=float)
    u_dofs_global = np.array([t[0], t[1], t[2], 0.0, 0.0, 0.0, t[0], t[1], t[2], 0.0, 0.0, 0.0], dtype=float)
    loads = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global))
    assert np.allclose(loads, np.zeros(12), atol=1e-12)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation
    """
    import numpy as np
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8e-06
    I_z = 4e-06
    J = 1.5e-05
    L = 2.5
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    u_axial = np.zeros(12, dtype=float)
    u_axial[6] = 1.0
    loads_axial = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial))
    k_axial = E * A / L
    expected_axial = np.zeros(12, dtype=float)
    expected_axial[0] = -k_axial
    expected_axial[6] = +k_axial
    assert np.allclose(loads_axial, expected_axial, rtol=0.0, atol=1e-10)
    u_shear_y = np.zeros(12, dtype=float)
    u_shear_y[7] = 1.0
    loads_shear_y = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y))
    EIz = E * I_z
    expected_shear_y = np.zeros(12, dtype=float)
    expected_shear_y[1] = -12.0 * EIz / L ** 3
    expected_shear_y[5] = -6.0 * EIz / L ** 2
    expected_shear_y[7] = +12.0 * EIz / L ** 3
    expected_shear_y[11] = -6.0 * EIz / L ** 2
    assert np.allclose(loads_shear_y, expected_shear_y, rtol=0.0, atol=1e-10)
    u_torsion = np.zeros(12, dtype=float)
    u_torsion[9] = 1.0
    loads_torsion = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion))
    G = E / (2.0 * (1.0 + nu))
    k_torsion = G * J / L
    expected_torsion = np.zeros(12, dtype=float)
    expected_torsion[3] = -k_torsion
    expected_torsion[9] = +k_torsion
    assert np.allclose(loads_torsion, expected_torsion, rtol=0.0, atol=1e-10)

def test_superposition_linearity(fcn):
    """Verify linearity: f(ua + ub) == f(ua) + f(ub) for arbitrary displacement states."""
    import numpy as np
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8e-06
    I_z = 4e-06
    J = 1.5e-05
    L = 2.5
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    ua = np.array([0.1, -0.05, 0.02, 0.01, -0.02, 0.03, 0.2, 0.15, -0.08, -0.04, 0.06, -0.01], dtype=float)
    ub = np.array([-0.03, 0.07, -0.09, 0.02, 0.01, -0.02, 0.04, -0.06, 0.05, -0.03, 0.02, 0.04], dtype=float)
    fa = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua))
    fb = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ub))
    fsum = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub))
    assert np.allclose(fsum, fa + fb, rtol=1e-12, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: rotating coordinates, displacements, and local_z by a rigid global rotation should leave the local end-load vector unchanged."""
    import numpy as np

    def rotation_matrix(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        ax, ay, az = axis
        K = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], dtype=float)
        I = np.eye(3)
        R = I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
        return R
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8e-06
    I_z = 4e-06
    J = 1.5e-05
    L = 2.5
    local_z0 = np.array([0.0, 0.0, 1.0], dtype=float)
    ele_info0 = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z0}
    xi0, yi0, zi0 = (0.0, 0.0, 0.0)
    xj0, yj0, zj0 = (L, 0.0, 0.0)
    u1_t = np.array([-0.3, 0.4, 0.2], dtype=float)
    u1_r = np.array([0.1, -0.2, 0.05], dtype=float)
    u2_t = np.array([0.8, -0.6, 0.1], dtype=float)
    u2_r = np.array([-0.3, 0.25, -0.15], dtype=float)
    u_dofs0 = np.array([u1_t[0], u1_t[1], u1_t[2], u1_r[0], u1_r[1], u1_r[2], u2_t[0], u2_t[1], u2_t[2], u2_r[0], u2_r[1], u2_r[2]], dtype=float)
    f_local0 = np.asarray(fcn(ele_info0, xi0, yi0, zi0, xj0, yj0, zj0, u_dofs0))
    axis = np.array([0.3, -0.5, 0.8], dtype=float)
    angle = 1.1
    R = rotation_matrix(axis, angle)
    xi_vec = R @ np.array([xi0, yi0, zi0], dtype=float)
    xj_vec = R @ np.array([xj0, yj0, zj0], dtype=float)
    local_z_rot = R @ local_z0
    ele_info_rot = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot}
    u1_t_rot = R @ u1_t
    u1_r_rot = R @ u1_r
    u2_t_rot = R @ u2_t
    u2_r_rot = R @ u2_r
    u_dofs_rot = np.array([u1_t_rot[0], u1_t_rot[1], u1_t_rot[2], u1_r_rot[0], u1_r_rot[1], u1_r_rot[2], u2_t_rot[0], u2_t_rot[1], u2_t_rot[2], u2_r_rot[0], u2_r_rot[1], u2_r_rot[2]], dtype=float)
    f_local_rot = np.asarray(fcn(ele_info_rot, float(xi_vec[0]), float(xi_vec[1]), float(xi_vec[2]), float(xj_vec[0]), float(xj_vec[1]), float(xj_vec[2]), u_dofs_rot))
    assert np.allclose(f_local_rot, f_local0, rtol=1e-12, atol=1e-10)