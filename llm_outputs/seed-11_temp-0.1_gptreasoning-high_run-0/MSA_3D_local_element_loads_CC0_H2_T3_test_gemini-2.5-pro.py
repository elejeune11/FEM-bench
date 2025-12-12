def test_rigid_body_motion_zero_loads(fcn):
    """
    Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector.
    """
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (10.0, 0.0, 0.0)
    (dx, dy, dz) = (0.1, -0.2, 0.3)
    u_dofs_global = np.array([dx, dy, dz, 0, 0, 0, dx, dy, dz, 0, 0, 0])
    expected_loads = np.zeros(12)
    actual_loads = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(actual_loads, expected_loads, atol=1e-09)

def test_unit_responses_axial_shear_torsion(fcn):
    """
    Single stand-alone test covering three unit responses:
      (1) Axial unit extension
      (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
      (3) Unit torsional rotation
    """
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (10.0, 0.0, 0.0)
    L = 10.0
    (E, nu, A) = (ele_info['E'], ele_info['nu'], ele_info['A'])
    (I_z, J) = (ele_info['I_z'], ele_info['J'])
    G = E / (2 * (1 + nu))
    u_axial = np.zeros(12)
    u_axial[6] = 1.0
    expected_axial = np.zeros(12)
    expected_axial[0] = -E * A / L
    expected_axial[6] = E * A / L
    actual_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
    assert np.allclose(actual_axial, expected_axial)
    u_shear = np.zeros(12)
    u_shear[1] = 1.0
    expected_shear = np.zeros(12)
    expected_shear[1] = 12 * E * I_z / L ** 3
    expected_shear[5] = 6 * E * I_z / L ** 2
    expected_shear[7] = -12 * E * I_z / L ** 3
    expected_shear[11] = 6 * E * I_z / L ** 2
    actual_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
    assert np.allclose(actual_shear, expected_shear)
    u_torsion = np.zeros(12)
    u_torsion[3] = 1.0
    expected_torsion = np.zeros(12)
    expected_torsion[3] = G * J / L
    expected_torsion[9] = -G * J / L
    actual_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
    assert np.allclose(actual_torsion, expected_torsion)

def test_superposition_linearity(fcn):
    """
    Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds.
    """
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    (xi, yi, zi) = (1, 2, 3)
    (xj, yj, zj) = (11, 4, -2)
    np.random.seed(0)
    u_a = np.random.rand(12)
    u_b = np.random.rand(12)
    f_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_a)
    f_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_b)
    u_c = u_a + u_b
    f_c = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_c)
    f_sum = f_a + f_b
    assert np.allclose(f_c, f_sum)

def test_coordinate_invariance_global_rotation(fcn):
    """
    Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged.
    """
    import numpy as np
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    p_i = np.array([1.0, 2.0, 3.0])
    p_j = np.array([11.0, 4.0, -2.0])
    local_z_1 = np.array([0.0, 0.0, 1.0])
    np.random.seed(42)
    u_global_1 = np.random.rand(12)
    ele_info_1 = ele_info.copy()
    ele_info_1['local_z'] = local_z_1
    f_local_1 = fcn(ele_info_1, *p_i, *p_j, u_global_1)
    theta = np.pi / 3
    (c, s) = (np.cos(theta), np.sin(theta))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    p_i_rot = R @ p_i
    p_j_rot = R @ p_j
    local_z_2 = R @ local_z_1
    u1_trans_rot = R @ u_global_1[0:3]
    u1_rot_rot = R @ u_global_1[3:6]
    u2_trans_rot = R @ u_global_1[6:9]
    u2_rot_rot = R @ u_global_1[9:12]
    u_global_2 = np.concatenate([u1_trans_rot, u1_rot_rot, u2_trans_rot, u2_rot_rot])
    ele_info_2 = ele_info.copy()
    ele_info_2['local_z'] = local_z_2
    f_local_2 = fcn(ele_info_2, *p_i_rot, *p_j_rot, u_global_2)
    assert np.allclose(f_local_1, f_local_2)