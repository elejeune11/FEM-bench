def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector.
    """
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0005, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (5.0, 0.0, 0.0)
    translation = np.array([1.5, -2.0, 0.1])
    rotation = np.zeros(3)
    u_node = np.concatenate([translation, rotation])
    u_dofs_global = np.concatenate([u_node, u_node])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-09)

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation
    """
    L = 2.0
    E = 1000.0
    nu = 0.25
    G = E / (2 * (1 + nu))
    A = 0.1
    Iz = 0.02
    J = 0.05
    ele_info = {'E': E, 'nu': nu, 'A': A, 'I_y': 0.01, 'I_z': Iz, 'J': J, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (L, 0.0, 0.0)
    u_axial = np.zeros(12)
    u_axial[6] = 1.0
    f_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    axial_stiff = E * A / L
    assert np.isclose(f_axial[0], -axial_stiff)
    assert np.isclose(f_axial[6], axial_stiff)
    u_shear = np.zeros(12)
    u_shear[7] = 1.0
    f_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    k_shear = 12 * E * Iz / L ** 3
    k_moment = 6 * E * Iz / L ** 2
    assert np.isclose(f_shear[1], -k_shear)
    assert np.isclose(f_shear[5], -k_moment)
    assert np.isclose(f_shear[7], k_shear)
    assert np.isclose(f_shear[11], -k_moment)
    u_torsion = np.zeros(12)
    u_torsion[9] = 1.0
    f_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    torsional_stiff = G * J / L
    assert np.isclose(f_torsion[3], -torsional_stiff)
    assert np.isclose(f_torsion[9], torsional_stiff)

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds.
    """
    ele_info = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.005, 'I_y': 3e-05, 'I_z': 4e-05, 'J': 1e-05, 'local_z': np.array([0, 1, 0])}
    (xi, yi, zi) = (1.0, 2.0, 3.0)
    (xj, yj, zj) = (4.0, 6.0, 3.0)
    rng = np.random.default_rng(42)
    ua = rng.standard_normal(12) * 0.001
    ub = rng.standard_normal(12) * 0.001
    fa = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
    fb = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
    f_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
    assert np.allclose(f_combined, fa + fb, atol=1e-08)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """
    ele_info = {'E': 70000000000.0, 'nu': 0.33, 'A': 0.02, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0005, 'local_z': np.array([0, 0, 1])}
    p1 = np.array([0.0, 0.0, 0.0])
    p2 = np.array([2.0, 0.0, 0.0])
    u_global_ref = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.01, 0.02, 0.0, 0.1, 0.0, 0.05])
    loads_ref = fcn(ele_info, p1[0], p1[1], p1[2], p2[0], p2[1], p2[2], u_global_ref)
    R = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    p1_rot = R @ p1
    p2_rot = R @ p2
    local_z_rot = R @ ele_info['local_z']
    u1_trans = R @ u_global_ref[0:3]
    u1_rot = R @ u_global_ref[3:6]
    u2_trans = R @ u_global_ref[6:9]
    u2_rot = R @ u_global_ref[9:12]
    u_global_rot = np.concatenate([u1_trans, u1_rot, u2_trans, u2_rot])
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = local_z_rot
    loads_rot = fcn(ele_info_rot, p1_rot[0], p1_rot[1], p1_rot[2], p2_rot[0], p2_rot[1], p2_rot[2], u_global_rot)
    assert np.allclose(loads_rot, loads_ref, atol=1e-08)