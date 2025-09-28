def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (1.0, 0.0, 0.0)
    translation = np.array([0.5, -0.3, 0.7])
    u_dofs_global = np.array([translation[0], translation[1], translation[2], 0, 0, 0, translation[0], translation[1], translation[2], 0, 0, 0])
    with patch('beam_transformation_matrix_3D') as mock_transform, patch('local_elastic_stiffness_matrix_3D_beam') as mock_stiffness:
        mock_transform.return_value = np.eye(12)
        mock_stiffness.return_value = np.diag([1000000.0, 1000.0, 1000.0, 100.0, 10000.0, 10000.0, 1000000.0, 1000.0, 1000.0, 100.0, 10000.0, 10000.0])
        load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
        assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-10)

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (2.0, 0.0, 0.0)
    L = 2.0
    E = ele_info['E']
    A = ele_info['A']
    I_y = ele_info['I_y']
    I_z = ele_info['I_z']
    J = ele_info['J']
    G = E / (2 * (1 + ele_info['nu']))
    with patch('beam_transformation_matrix_3D') as mock_transform, patch('local_elastic_stiffness_matrix_3D_beam') as mock_stiffness:
        mock_transform.return_value = np.eye(12)
        K_local = np.zeros((12, 12))
        K_local[0, 0] = K_local[6, 6] = E * A / L
        K_local[0, 6] = K_local[6, 0] = -E * A / L
        K_local[3, 3] = K_local[9, 9] = G * J / L
        K_local[3, 9] = K_local[9, 3] = -G * J / L
        K_local[1, 1] = K_local[7, 7] = 12 * E * I_z / L ** 3
        K_local[1, 7] = K_local[7, 1] = -12 * E * I_z / L ** 3
        K_local[1, 5] = K_local[5, 1] = 6 * E * I_z / L ** 2
        K_local[1, 11] = K_local[11, 1] = 6 * E * I_z / L ** 2
        K_local[7, 5] = K_local[5, 7] = -6 * E * I_z / L ** 2
        K_local[7, 11] = K_local[11, 7] = -6 * E * I_z / L ** 2
        K_local[5, 5] = K_local[11, 11] = 4 * E * I_z / L
        K_local[5, 11] = K_local[11, 5] = 2 * E * I_z / L
        K_local[2, 2] = K_local[8, 8] = 12 * E * I_y / L ** 3
        K_local[2, 8] = K_local[8, 2] = -12 * E * I_y / L ** 3
        K_local[2, 4] = K_local[4, 2] = -6 * E * I_y / L ** 2
        K_local[2, 10] = K_local[10, 2] = -6 * E * I_y / L ** 2
        K_local[8, 4] = K_local[4, 8] = 6 * E * I_y / L ** 2
        K_local[8, 10] = K_local[10, 8] = 6 * E * I_y / L ** 2
        K_local[4, 4] = K_local[10, 10] = 4 * E * I_y / L
        K_local[4, 10] = K_local[10, 4] = 2 * E * I_y / L
        mock_stiffness.return_value = K_local
        u_axial = np.zeros(12)
        u_axial[6] = 1.0
        load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
        assert np.abs(load_axial[0] + E * A / L) < 1e-10
        assert np.abs(load_axial[6] - E * A / L) < 1e-10
        u_shear = np.zeros(12)
        u_shear[7] = 1.0
        load_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
        assert np.abs(load_shear[1] + 12 * E * I_z / L ** 3) < 1e-10
        assert np.abs(load_shear[7] - 12 * E * I_z / L ** 3) < 1e-10
        u_torsion = np.zeros(12)
        u_torsion[9] = 1.0
        load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
        assert np.abs(load_torsion[3] + G * J / L) < 1e-10
        assert np.abs(load_torsion[9] - G * J / L) < 1e-10

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    ele_info = {'E': 210000000000.0, 'nu': 0.28, 'A': 0.015, 'I_y': 2e-05, 'I_z': 3e-05, 'J': 4e-05, 'local_z': np.array([0, 1, 0])}
    (xi, yi, zi) = (1.0, 2.0, 3.0)
    (xj, yj, zj) = (4.0, 5.0, 6.0)
    np.random.seed(42)
    ua = np.random.randn(12) * 0.01
    ub = np.random.randn(12) * 0.01
    with patch('beam_transformation_matrix_3D') as mock_transform, patch('local_elastic_stiffness_matrix_3D_beam') as mock_stiffness:
        mock_transform.return_value = np.eye(12)
        mock_stiffness.return_value = np.random.randn(12, 12)
        mock_stiffness.return_value = mock_stiffness.return_value + mock_stiffness.return_value.T
        mock_stiffness.return_value = mock_stiffness.return_value + np.eye(12) * 10
        load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
        load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
        load_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua + ub)
        assert np.allclose(load_combined, load_a + load_b, rtol=1e-10)

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    ele_info = {'E': 195000000000.0, 'nu': 0.31, 'A': 0.02, 'I_y': 1.5e-05, 'I_z': 2.5e-05, 'J': 3.5e-05, 'local_z': np.array([0, 0, 1])}
    (xi, yi, zi) = (0.0, 0.0, 0.0)
    (xj, yj, zj) = (3.0, 0.0, 0.0)
    np.random.seed(123)
    u_dofs_global = np.random.randn(12) * 0.005
    theta = np.pi / 6
    phi = np.pi / 4
    R = np.array([[np.cos(theta) * np.cos(phi), -np.sin(phi), np.sin(theta) * np.cos(phi)], [np.cos(theta) * np.sin(phi), np.cos(phi), np.sin(theta) * np.sin(phi)], [-np.sin(theta), 0, np.cos(theta)]])
    coords_i = np.array([xi, yi, zi])
    coords_j = np.array([xj, yj, zj])
    coords_i_rot = R @ coords_i
    coords_j_rot = R @ coords_j
    u_dofs_rot = np.zeros(12)
    u_dofs_rot[0:3] = R @ u_dofs_global[0:3]
    u_dofs_rot[3:6] = R @ u_dofs_global[3:6]
    u_dofs_rot[6:9] = R @ u_dofs_global[6:9]
    u_dofs_rot[9:12] = R @ u_dofs_global[9:12]
    ele_info_rot = ele_info.copy()
    ele_info_rot['local_z'] = R @ ele_info['local_z']
    with patch('beam_transformation_matrix_3D') as mock_transform, patch('local_elastic_stiffness_matrix_3D_beam') as mock_stiffness:
        Gamma1 = np.eye(12)
        Gamma1[0:3, 0:3] = np.array([[1, 0, 0], [0, 0.8, 0.6], [0, -0.6, 0.8]])
        Gamma1[3:6, 3:6] = Gamma1[0:3, 0:3]
        Gamma1[6:9, 6:9] = Gamma1[0:3, 0:3]
        Gamma1[9:12, 9:12] = Gamma1[0:3, 0:3]
        Gamma2 = np.eye(12)
        Gamma2[0:3, 0:3] = np.array([[1, 0, 0], [0, 0.7, 0.714], [0, -0.714, 0.7]])
        Gamma2[3:6, 3:6] = Gamma2[0:3, 0:3]
        Gamma2[6:9, 6:9] = Gamma2[0:3, 0:3]
        Gamma2[9:12, 9:12] = Gamma2[0:3, 0:3]
        K_local = np.diag([10000000.0, 10000.0, 10000.0, 1000.0, 100000.0, 100000.0, 10000000.0, 10000.0, 10000.0, 1000.0, 100000.0, 100000.0])
        mock_transform.side_effect = [Gamma1, Gamma2]
        mock_stiffness.return_value = K_local
        load_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
        load_rotated = fcn(ele_info_rot, coords_i_rot[0], coords_i_rot[1], coords_i_rot[2], coords_j_rot[0], coords_j_rot[1], coords_j_rot[2], u_dofs_rot)
        assert np.allclose(load_original, load_rotated, rtol=1e-08)