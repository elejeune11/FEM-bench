def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 2e-06
    I_z = 3e-06
    J = 1e-05
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0.0, 0.0, 1.0]}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    u_tr = [1.2, -0.7, 0.4]
    r = [0.0, 0.0, 0.0]
    u_dofs_global = [u_tr[0], u_tr[1], u_tr[2], r[0], r[1], r[2], u_tr[0], u_tr[1], u_tr[2], r[0], r[1], r[2]]
    load = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    G = E / (2.0 * (1.0 + nu))
    k_scale = max(E * A / L, G * J / L, 12.0 * E * I_y / L ** 3, 12.0 * E * I_z / L ** 3)
    tol = 1e-09 * k_scale
    assert len(load) == 12
    for val in load:
        assert abs(val) <= tol

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
    L = 2.0
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0.0, 0.0, 1.0]}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    G = E / (2.0 * (1.0 + nu))
    u_axial = [0.0] * 12
    u_axial[0] = 0.0
    u_axial[6] = 1.0
    load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    k_ax = E * A / L
    tol_ax = 1e-12 * max(1.0, k_ax)
    assert abs(load_axial[0] - -k_ax) <= tol_ax
    assert abs(load_axial[6] - k_ax) <= tol_ax
    for idx in [1, 2, 3, 4, 5, 7, 8, 9, 10, 11]:
        assert abs(load_axial[idx]) <= tol_ax
    u_shear = [0.0] * 12
    u_shear[1] = 0.0
    u_shear[7] = 1.0
    load_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    EIz = E * I_z
    Fy = 12.0 * EIz / L ** 3
    Mz = 6.0 * EIz / L ** 2
    tol_sh = 1e-12 * max(1.0, Fy, Mz)
    assert abs(load_shear[1] - -Fy) <= tol_sh
    assert abs(load_shear[5] - -Mz) <= tol_sh
    assert abs(load_shear[7] - Fy) <= tol_sh
    assert abs(load_shear[11] - -Mz) <= tol_sh
    for idx in [0, 2, 3, 4, 6, 8, 9, 10]:
        assert abs(load_shear[idx]) <= tol_sh
    u_torsion = [0.0] * 12
    u_torsion[3] = 0.0
    u_torsion[9] = 1.0
    load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    k_t = G * J / L
    tol_t = 1e-12 * max(1.0, k_t)
    assert abs(load_torsion[3] - -k_t) <= tol_t
    assert abs(load_torsion[9] - k_t) <= tol_t
    for idx in [0, 1, 2, 4, 5, 6, 7, 8, 10, 11]:
        assert abs(load_torsion[idx]) <= tol_t

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a combined displacement state (ua + ub)
    equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds.
    """
    E = 200000000000.0
    nu = 0.28
    A = 0.02
    I_y = 1.8e-06
    I_z = 2.5e-06
    J = 4e-06
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0.0, 0.0, 1.0]}
    (xi, yi, zi) = (1.0, -2.0, 0.5)
    (xj, yj, zj) = (2.1, 0.7, 3.3)
    ua = [0.1, -0.2, 0.05, 0.01, 0.02, -0.03, -0.04, 0.03, -0.06, 0.02, -0.01, 0.005]
    ub = [-0.02, 0.04, 0.1, -0.02, 0.0, 0.01, 0.06, -0.07, 0.02, -0.03, 0.04, -0.02]
    uab = [ua[i] + ub[i] for i in range(12)]
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    fs = fcn(ele_info, xi, yi, zi, xj, yj, zj, uab)
    assert len(fa) == 12 and len(fb) == 12 and (len(fs) == 12)
    for i in range(12):
        rhs = fa[i] + fb[i]
        denom = abs(fs[i]) + abs(rhs) + 1.0
        tol = 1e-09 * denom
        assert abs(fs[i] - rhs) <= tol

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z) by a rigid global rotation R,
    the local internal end-load vector should be unchanged.
    """
    E = 190000000000.0
    nu = 0.31
    A = 0.015
    I_y = 2.2e-06
    I_z = 1.7e-06
    J = 3.5e-06

    def Rz90(v):
        return [-v[1], v[0], v[2]]
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0.0, 0.0, 1.0]}
    (xi, yi, zi) = (0.5, -1.0, 2.0)
    (xj, yj, zj) = (1.3, 2.2, 3.7)
    u_dofs = [0.12, -0.06, 0.2, -0.03, 0.02, 0.04, -0.15, 0.07, -0.11, 0.05, -0.02, 0.01]
    f_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs)
    (xi_r, yi_r, zi_r) = Rz90([xi, yi, zi])
    (xj_r, yj_r, zj_r) = Rz90([xj, yj, zj])
    local_z_r = Rz90(ele_info['local_z'])
    u_r = [0.0] * 12
    t1 = [u_dofs[0], u_dofs[1], u_dofs[2]]
    r1 = [u_dofs[3], u_dofs[4], u_dofs[5]]
    t1r = Rz90(t1)
    r1r = Rz90(r1)
    (u_r[0], u_r[1], u_r[2]) = (t1r[0], t1r[1], t1r[2])
    (u_r[3], u_r[4], u_r[5]) = (r1r[0], r1r[1], r1r[2])
    t2 = [u_dofs[6], u_dofs[7], u_dofs[8]]
    r2 = [u_dofs[9], u_dofs[10], u_dofs[11]]
    t2r = Rz90(t2)
    r2r = Rz90(r2)
    (u_r[6], u_r[7], u_r[8]) = (t2r[0], t2r[1], t2r[2])
    (u_r[9], u_r[10], u_r[11]) = (r2r[0], r2r[1], r2r[2])
    ele_info_rot = {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_r}
    f_local_rot = fcn(ele_info_rot, xi_r, yi_r, zi_r, xj_r, yj_r, zj_r, u_r)
    assert len(f_local) == 12 and len(f_local_rot) == 12
    for i in range(12):
        denom = abs(f_local[i]) + abs(f_local_rot[i]) + 1.0
        tol = 1e-09 * denom
        assert abs(f_local[i] - f_local_rot[i]) <= tol