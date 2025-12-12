def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.array([1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0])
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    assert np.allclose(load_dofs_local, np.zeros(12))

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses: (1) Axial unit extension, (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations, and (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
    K_local = local_elastic_stiffness_matrix_3D_beam(ele_info['E'], ele_info['nu'], ele_info['A'], L, ele_info['I_y'], ele_info['I_z'], ele_info['J'])
    Gamma = beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj)
    u_dofs_global_axial = np.dot(Gamma.T, np.array([1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]))
    u_dofs_global_shear = np.dot(Gamma.T, np.array([0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0]))
    u_dofs_global_torsion = np.dot(Gamma.T, np.array([0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0]))
    load_dofs_local_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_axial)
    load_dofs_local_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_shear)
    load_dofs_local_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_torsion)
    assert np.isclose(load_dofs_local_axial[0], -K_local[0, 0])
    assert np.isclose(load_dofs_local_axial[6], K_local[0, 0])
    assert np.isclose(load_dofs_local_shear[1], -K_local[1, 1])
    assert np.isclose(load_dofs_local_shear[7], K_local[1, 1])
    assert np.isclose(load_dofs_local_torsion[3], -K_local[3, 3])
    assert np.isclose(load_dofs_local_torsion[9], K_local[3, 3])

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a combined displacement state (ua + ub) equals the sum of the individual responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global_a = np.random.rand(12)
    u_dofs_global_b = np.random.rand(12)
    u_dofs_global_sum = u_dofs_global_a + u_dofs_global_b
    load_dofs_local_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_a)
    load_dofs_local_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_b)
    load_dofs_local_sum = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global_sum)
    assert np.allclose(load_dofs_local_sum, load_dofs_local_a + load_dofs_local_b)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance: If we rotate the entire configuration (coords, displacements, and local_z) by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0, 0, 0)
    (xj, yj, zj) = (1, 0, 0)
    u_dofs_global = np.random.rand(12)
    R = np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]])
    (xi_rot, yi_rot, zi_rot) = np.dot(R, [xi, yi, zi])
    (xj_rot, yj_rot, zj_rot) = np.dot(R, [xj, yj, zj])
    u_dofs_global_rot = np.dot(np.kron(np.eye(4), R), u_dofs_global)
    local_z_rot = np.dot(R, ele_info['local_z'])
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = local_z_rot
    load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
    load_dofs_local_rot = fcn(ele_info_rot, xi_rot, yi_rot, zi_rot, xj_rot, yj_rot, zj_rot, u_dofs_global_rot)
    assert np.allclose(load_dofs_local, load_dofs_local_rot)