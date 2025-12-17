def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a straight 3D beam element produces
    zero internal forces and moments in the local load vector.
    """
    import numpy as np
    E = 70000000000.0
    nu = 0.3
    A = 0.001
    I_y = 2e-06
    I_z = 3e-06
    J = 1e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    L = 2.0
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    t = [0.5, -1.2, 3.3]
    u_dofs_global = np.array([t[0], t[1], t[2], 0.0, 0.0, 0.0, t[0], t[1], t[2], 0.0, 0.0, 0.0])
    loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(loads, np.zeros(12), atol=1e-10, rtol=0.0)

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
      (1) Axial unit extension: expect +/- EA/L axial forces only
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations:
          expect Fy = +/- 12*E*Iz/L^3 and Mz = -6*E*Iz/L^2 at both ends
      (3) Unit torsional rotation: expect Mx = +/- GJ/L only
    """
    import numpy as np
    E = 70000000000.0
    nu = 0.3
    A = 0.001
    I_y = 2e-06
    I_z = 3e-06
    J = 1e-06
    G = E / (2.0 * (1.0 + nu))
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}
    L = 2.0
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    loads_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    k_axial = E * A / L
    assert np.isclose(loads_axial[0], -k_axial, rtol=1e-12, atol=1e-10)
    assert np.isclose(loads_axial[6], +k_axial, rtol=1e-12, atol=1e-10)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[0, 6]] = False
    assert np.allclose(loads_axial[mask_other], 0.0, atol=1e-09)
    u_shear_y = np.zeros(12)
    u_shear_y[1] = 0.0
    u_shear_y[7] = 1.0
    loads_shear_y = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear_y)
    Fy_mag = 12.0 * E * I_z / L ** 3
    Mz_mag = 6.0 * E * I_z / L ** 2
    assert np.isclose(loads_shear_y[1], -Fy_mag, rtol=1e-12, atol=1e-10)
    assert np.isclose(loads_shear_y[7], +Fy_mag, rtol=1e-12, atol=1e-10)
    assert np.isclose(loads_shear_y[5], -Mz_mag, rtol=1e-12, atol=1e-10)
    assert np.isclose(loads_shear_y[11], -Mz_mag, rtol=1e-12, atol=1e-10)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[1, 5, 7, 11]] = False
    assert np.allclose(loads_shear_y[mask_other], 0.0, atol=1e-08)
    u_torsion = np.zeros(12)
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    loads_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    k_torsion = G * J / L
    assert np.isclose(loads_torsion[3], -k_torsion, rtol=1e-12, atol=1e-10)
    assert np.isclose(loads_torsion[9], +k_torsion, rtol=1e-12, atol=1e-10)
    mask_other = np.ones(12, dtype=bool)
    mask_other[[3, 9]] = False
    assert np.allclose(loads_torsion[mask_other], 0.0, atol=1e-09)

def test_superposition_linearity(fcn):
    """
    Verify linearity: for any two displacement states ua and ub,
    f(ua + ub) equals f(ua) + f(ub) within numerical tolerance.
    """
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.29, 'A': 0.0025, 'I_y': 4e-06, 'I_z': 5e-06, 'J': 3.1e-06, 'local_z': np.array([0.0, 0.0, 1.0])}
    L = 3.0
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    ua = np.array([0.1, -0.03, 0.05, 0.01, -0.02, 0.04, 0.07, -0.01, -0.02, -0.03, 0.06, -0.02])
    ub = np.array([-0.02, 0.09, -0.04, -0.01, 0.03, 0.02, 0.05, 0.02, 0.01, 0.02, -0.04, 0.03])
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fab = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert np.allclose(fab, fa + fb, rtol=1e-12, atol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance:
    If we rotate the entire configuration (coordinates, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """
    import numpy as np
    E = 190000000000.0
    nu = 0.28
    A = 0.0018
    I_y = 3.6e-06
    I_z = 2.7e-06
    J = 2.2e-06
    local_z = np.array([0.0, 0.0, 1.0])
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}
    L = 2.5
    xi, yi, zi = (0.0, 0.0, 0.0)
    xj, yj, zj = (L, 0.0, 0.0)
    u = np.array([0.2, -0.1, 0.05, 0.03, -0.04, 0.02, 0.3, -0.05, -0.04, -0.02, 0.06, -0.01])
    f_ref = fcn(ele_info, xi, yi, zi, xj, yj, zj, u)
    angle = 0.41
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    ri = np.array([xi, yi, zi])
    rj = np.array([xj, yj, zj])
    ri_rot = R @ ri
    rj_rot = R @ rj
    xi_r, yi_r, zi_r = ri_rot.tolist()
    xj_r, yj_r, zj_r = rj_rot.tolist()
    local_z_rot = R @ local_z
    ele_info_rot = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rot}

    def rotate_u12(u12):
        t1 = R @ u12[0:3]
        r1 = R @ u12[3:6]
        t2 = R @ u12[6:9]
        r2 = R @ u12[9:12]
        return np.hstack((t1, r1, t2, r2))
    u_rot = rotate_u12(u)
    f_rot = fcn(ele_info_rot, xi_r, yi_r, zi_r, xj_r, yj_r, zj_r, u_rot)
    assert np.allclose(f_rot, f_ref, rtol=1e-11, atol=1e-10)