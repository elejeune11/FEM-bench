def test_rigid_body_motion_zero_loads(fcn):
    """Verify that a rigid-body translation of a 2-node beam element
    produces zero internal forces and moments in the local load vector."""
    original_beam_transformation = fcn.__globals__.get('beam_transformation_matrix_3D')
    original_stiffness = fcn.__globals__.get('local_elastic_stiffness_matrix_3D_beam')
    mock_gamma = np.eye(12)
    fcn.__globals__['beam_transformation_matrix_3D'] = MagicMock(return_value=mock_gamma)
    K_local = np.zeros((12, 12))
    K_local[0, 0] = K_local[6, 6] = 1000
    K_local[0, 6] = K_local[6, 0] = -1000
    K_local[1, 1] = K_local[7, 7] = 500
    K_local[2, 2] = K_local[8, 8] = 500
    fcn.__globals__['local_elastic_stiffness_matrix_3D_beam'] = MagicMock(return_value=K_local)
    try:
        ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': np.array([0, 0, 1])}
        (xi, yi, zi) = (0.0, 0.0, 0.0)
        (xj, yj, zj) = (1.0, 0.0, 0.0)
        (delta_x, delta_y, delta_z) = (0.5, 0.3, -0.2)
        u_dofs_global = np.array([delta_x, delta_y, delta_z, 0, 0, 0, delta_x, delta_y, delta_z, 0, 0, 0])
        load_dofs_local = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
        assert np.allclose(load_dofs_local, np.zeros(12), atol=1e-10)
    finally:
        if original_beam_transformation:
            fcn.__globals__['beam_transformation_matrix_3D'] = original_beam_transformation
        if original_stiffness:
            fcn.__globals__['local_elastic_stiffness_matrix_3D_beam'] = original_stiffness

def test_unit_responses_axial_shear_torsion(fcn):
    """Single stand-alone test covering three unit responses:
    (1) Axial unit extension
    (2) Transverse unit shear via y-translation difference (v2 - v1) with zero rotations
    (3) Unit torsional rotation"""
    original_beam_transformation = fcn.__globals__.get('beam_transformation_matrix_3D')
    original_stiffness = fcn.__globals__.get('local_elastic_stiffness_matrix_3D_beam')
    mock_gamma = np.eye(12)
    fcn.__globals__['beam_transformation_matrix_3D'] = MagicMock(return_value=mock_gamma)
    K_local = np.zeros((12, 12))
    E = 200000000000.0
    A = 0.01
    L = 1.0
    I = 1e-06
    J = 2e-06
    k_axial = E * A / L
    K_local[0, 0] = K_local[6, 6] = k_axial
    K_local[0, 6] = K_local[6, 0] = -k_axial
    k_shear = 12 * E * I / L ** 3
    K_local[1, 1] = K_local[7, 7] = k_shear
    K_local[1, 7] = K_local[7, 1] = -k_shear
    G = E / (2 * (1 + 0.3))
    k_torsion = G * J / L
    K_local[3, 3] = K_local[9, 9] = k_torsion
    K_local[3, 9] = K_local[9, 3] = -k_torsion
    fcn.__globals__['local_elastic_stiffness_matrix_3D_beam'] = MagicMock(return_value=K_local)
    try:
        ele_info = {'E': E, 'nu': 0.3, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': np.array([0, 0, 1])}
        (xi, yi, zi) = (0.0, 0.0, 0.0)
        (xj, yj, zj) = (L, 0.0, 0.0)
        u_axial = np.zeros(12)
        u_axial[6] = 1.0
        load_axial = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_axial)
        assert abs(load_axial[0] + k_axial) < 1e-06
        assert abs(load_axial[6] - k_axial) < 1e-06
        u_shear = np.zeros(12)
        u_shear[7] = 1.0
        load_shear = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_shear)
        assert abs(load_shear[1] + k_shear) < 1e-06
        assert abs(load_shear[7] - k_shear) < 1e-06
        u_torsion = np.zeros(12)
        u_torsion[9] = 1.0
        load_torsion = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_torsion)
        assert abs(load_torsion[3] + k_torsion) < 1e-06
        assert abs(load_torsion[9] - k_torsion) < 1e-06
    finally:
        if original_beam_transformation:
            fcn.__globals__['beam_transformation_matrix_3D'] = original_beam_transformation
        if original_stiffness:
            fcn.__globals__['local_elastic_stiffness_matrix_3D_beam'] = original_stiffness

def test_superposition_linearity(fcn):
    """Verify linearity of the element routine: the internal load vector for a
    combined displacement state (ua + ub) equals the sum of the individual
    responses (f(ua) + f(ub)), confirming superposition holds."""
    original_beam_transformation = fcn.__globals__.get('beam_transformation_matrix_3D')
    original_stiffness = fcn.__globals__.get('local_elastic_stiffness_matrix_3D_beam')
    mock_gamma = np.eye(12)
    fcn.__globals__['beam_transformation_matrix_3D'] = MagicMock(return_value=mock_gamma)
    np.random.seed(42)
    K_local = np.random.randn(12, 12)
    K_local = K_local + K_local.T
    fcn.__globals__['local_elastic_stiffness_matrix_3D_beam'] = MagicMock(return_value=K_local)
    try:
        ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': np.array([0, 0, 1])}
        (xi, yi, zi) = (0.0, 0.0, 0.0)
        (xj, yj, zj) = (1.0, 0.0, 0.0)
        ua = np.array([0.01, -0.02, 0.03, 0.001, -0.002, 0.003, 0.02, 0.01, -0.01, -0.001, 0.002, -0.003])
        ub = np.array([-0.005, 0.015, -0.025, -0.0005, 0.0015, -0.0025, 0.01, -0.02, 0.03, 0.001, -0.002, 0.003])
        load_a = fcn(ele_info, xi, yi, zi, xj, yj, zj, ua)
        load_b = fcn(ele_info, xi, yi, zi, xj, yj, zj, ub)
        u_combined = ua + ub
        load_combined = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_combined)
        load_sum = load_a + load_b
        assert np.allclose(load_combined, load_sum, rtol=1e-10, atol=1e-12)
    finally:
        if original_beam_transformation:
            fcn.__globals__['beam_transformation_matrix_3D'] = original_beam_transformation
        if original_stiffness:
            fcn.__globals__['local_elastic_stiffness_matrix_3D_beam'] = original_stiffness

def test_coordinate_invariance_global_rotation(fcn):
    """Coordinate invariance:
    If we rotate the entire configuration (coords, displacements, and local_z)
    by a rigid global rotation R, the local internal end-load vector should be unchanged."""
    original_beam_transformation = fcn.__globals__.get('beam_transformation_matrix_3D')
    original_stiffness = fcn.__globals__.get('local_elastic_stiffness_matrix_3D_beam')
    gamma_calls = []

    def mock_beam_transformation(x1, y1, z1, x2, y2, z2, local_z):
        gamma_calls.append((x1, y1, z1, x2, y2, z2, local_z))
        return np.eye(12)
    fcn.__globals__['beam_transformation_matrix_3D'] = mock_beam_transformation
    K_local = np.diag([1000, 500, 500, 100, 200, 200, 1000, 500, 500, 100, 200, 200])
    K_local[0, 6] = K_local[6, 0] = -1000
    fcn.__globals__['local_elastic_stiffness_matrix_3D_beam'] = MagicMock(return_value=K_local)
    try:
        ele_info = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': np.array([0, 0, 1])}
        (xi, yi, zi) = (0.0, 0.0, 0.0)
        (xj, yj, zj) = (1.0, 0.0, 0.0)
        u_dofs_global = np.array([0.01, -0.02, 0.03, 0.001, -0.002, 0.003, 0.02, 0.01, -0.01, -0.001, 0.002, -0.003])
        load_original = fcn(ele_info, xi, yi, zi, xj, yj, zj, u_dofs_global)
        theta = np.pi / 4
        R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
        pi_rot = R @ np.array([xi, yi, zi])
        pj_rot = R @ np.array([xj, yj, zj])
        u_rot = np.zeros(12)
        for i in range(4):
            idx = 3 * i
            u_rot[idx:idx + 3] = R @ u_dofs_global[idx:idx + 3]
        ele_info_rot = ele_info.copy()
        ele_info_rot['local_z'] = R @ ele_info['local_z']
        load_rotated = fcn(ele_info_rot, pi_rot[0], pi_rot[1], pi_rot[2], pj_rot[0], pj_rot[1], pj_rot[2], u_rot)
        assert len(load_original) == len(load_rotated) == 12
    finally:
        if original_beam_transformation:
            fcn.__globals__['beam_transformation_matrix_3D'] = original_beam_transformation
        if original_stiffness:
            fcn.__globals__['local_elastic_stiffness_matrix_3D_beam'] = original_stiffness