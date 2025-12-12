def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    import numpy as np
    E = 210000000000.0
    nu = 0.3
    ele_info = {'E': E, 'nu': nu, 'A': 0.01, 'I_y': 1e-06, 'I_z': 2e-06, 'J': 5e-07, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    T = np.array([0.35, -0.12, 0.7], dtype=float)
    rot_zero = np.zeros(3, dtype=float)
    u_dofs_global = np.hstack((T, rot_zero, T, rot_zero))
    loads_local = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global))
    assert loads_local.shape == (12,)
    assert np.allclose(loads_local, np.zeros(12), atol=1e-08, rtol=1e-08)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    import numpy as np
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 2.0
    I_z = 3.0
    J = 0.5
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    L = float(np.linalg.norm(np.array([xj - xi, yj - yi, zj - zi])))
    G = E / (2.0 * (1.0 + nu))
    K = np.zeros((12, 12), dtype=float)
    k_ax = E * A / L
    K[0, 0] = k_ax
    K[0, 6] = -k_ax
    K[6, 0] = -k_ax
    K[6, 6] = k_ax
    k_tor = G * J / L
    K[3, 3] = k_tor
    K[3, 9] = -k_tor
    K[9, 3] = -k_tor
    K[9, 9] = k_tor
    k_bz_factor = E * I_z / L ** 3
    k_b = np.array([[12.0, 6.0 * L, -12.0, 6.0 * L], [6.0 * L, 4.0 * L ** 2, -6.0 * L, 2.0 * L ** 2], [-12.0, -6.0 * L, 12.0, -6.0 * L], [6.0 * L, 2.0 * L ** 2, -6.0 * L, 4.0 * L ** 2]], dtype=float)
    idx_bz = [1, 5, 7, 11]
    for ii in range(4):
        for jj in range(4):
            K[idx_bz[ii], idx_bz[jj]] = k_bz_factor * k_b[ii, jj]
    k_by_factor = E * I_y / L ** 3
    idx_by = [2, 4, 8, 10]
    for ii in range(4):
        for jj in range(4):
            K[idx_by[ii], idx_by[jj]] = k_by_factor * k_b[ii, jj]
    u_axial = np.zeros(12, dtype=float)
    u_axial[6] = 1.0
    expected_axial = K @ u_axial
    out_axial = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial))
    assert np.allclose(out_axial, expected_axial, atol=1e-08, rtol=1e-08)
    u_shear = np.zeros(12, dtype=float)
    u_shear[7] = 1.0
    expected_shear = K @ u_shear
    out_shear = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear))
    assert np.allclose(out_shear, expected_shear, atol=1e-08, rtol=1e-08)
    u_torsion = np.zeros(12, dtype=float)
    u_torsion[9] = 1.0
    expected_torsion = K @ u_torsion
    out_torsion = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion))
    assert np.allclose(out_torsion, expected_torsion, atol=1e-08, rtol=1e-08)

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a combined displacement state (ua + ub) equals the sum of the individual responses (f(ua) + f(ub))."""
    import numpy as np
    E = 210000000000.0
    nu = 0.25
    ele_info = {'E': E, 'nu': nu, 'A': 0.02, 'I_y': 1.5e-05, 'I_z': 2e-05, 'J': 1e-05, 'local_z': np.array([0.0, 0.0, 1.0])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    ua = np.array([0.12, -0.07, 0.03, 0.01, 0.02, -0.015, 0.05, 0.06, -0.02, -0.005, 0.004, 0.02], dtype=float)
    ub = np.array([-0.08, 0.03, 0.04, -0.012, 0.01, 0.009, 0.02, -0.03, 0.05, 0.006, -0.007, -0.01], dtype=float)
    fa = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua))
    fb = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ub))
    fab = np.asarray(fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub))
    assert np.allclose(fab, fa + fb, atol=1e-08, rtol=1e-08)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    import numpy as np
    E = 200000000000.0
    nu = 0.33
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)
    ele_info = {'E': E, 'nu': nu, 'A': 0.015, 'I_y': 1e-06, 'I_z': 2e-06, 'J': 3e-07, 'local_z': local_z}
    xi = np.array([0.0, 0.0, 0.0], dtype=float)
    xj = np.array([1.5, 0.0, 0.0], dtype=float)
    u1 = np.array([0.05, -0.02, 0.03], dtype=float)
    r1 = np.array([0.002, -0.004, 0.001], dtype=float)
    u2 = np.array([-0.01, 0.03, -0.02], dtype=float)
    r2 = np.array([-0.003, 0.005, 0.004], dtype=float)
    u_global = np.hstack((u1, r1, u2, r2))
    f_ref = np.asarray(fcn(ele_info, xi[0], xi[1], xi[2], xj[0], xj[1], xj[2], u_global))
    axis = np.array([1.0, 1.0, 0.5], dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = 0.743
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]], dtype=float)
    R = np.eye(3) * np.cos(theta) + (1.0 - np.cos(theta)) * np.outer(axis, axis) + np.sin(theta) * K
    xi_r = R @ xi
    xj_r = R @ xj
    local_z_r = R @ local_z
    u1_r = R @ u1
    r1_r = R @ r1
    u2_r = R @ u2
    r2_r = R @ r2
    u_global_r = np.hstack((u1_r, r1_r, u2_r, r2_r))
    ele_info_r = {'E': E, 'nu': nu, 'A': 0.015, 'I_y': 1e-06, 'I_z': 2e-06, 'J': 3e-07, 'local_z': local_z_r}
    f_rot = np.asarray(fcn(ele_info_r, xi_r[0], xi_r[1], xi_r[2], xj_r[0], xj_r[1], xj_r[2], u_global_r))
    assert np.allclose(f_ref, f_rot, atol=1e-08, rtol=1e-08)