def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.1, 'local_z': None}]

    def mock_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        return np.eye(12)

    def mock_compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        return np.array([1.0, 0.5, 0.0, 0.2, 0.3, 0.1, -1.0, -0.5, 0.0, -0.2, -0.3, -0.1])

    def mock_local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        return np.random.rand(12, 12) * 0.1
    with patch('__main__.beam_transformation_matrix_3D', mock_beam_transformation_matrix_3D), patch('__main__.compute_local_element_loads_beam_3D', mock_compute_local_element_loads_beam_3D), patch('__main__.local_geometric_stiffness_matrix_3D_beam', mock_local_geometric_stiffness_matrix_3D_beam):
        u_zero = np.zeros(18)
        with patch('__main__.compute_local_element_loads_beam_3D', return_value=np.zeros(12)), patch('__main__.local_geometric_stiffness_matrix_3D_beam', return_value=np.zeros((12, 12))):
            K_zero = fcn(node_coords, elements, u_zero)
            assert np.allclose(K_zero, 0.0)
        u_test = np.random.rand(18) * 0.1
        K = fcn(node_coords, elements, u_test)
        assert np.allclose(K, K.T, atol=1e-12)
        scale = 2.5
        u_scaled = scale * u_test
        K_scaled = fcn(node_coords, elements, u_scaled)
        K_expected = scale * K
        assert np.allclose(K_scaled, K_expected, rtol=1e-10)
        u1 = np.random.rand(18) * 0.1
        u2 = np.random.rand(18) * 0.1
        K1 = fcn(node_coords, elements, u1)
        K2 = fcn(node_coords, elements, u2)
        K_sum = fcn(node_coords, elements, u1 + u2)
        assert np.allclose(K_sum, K1 + K2, rtol=1e-10)
        elements_reordered = [elements[1], elements[0]]
        K_reordered = fcn(node_coords, elements_reordered, u_test)
        assert np.allclose(K, K_reordered, rtol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [1.5, 1.0, 0.5]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.1, 'local_z': [0, 0, 1]}]
    u_global = np.random.rand(18) * 0.1
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    T = np.zeros((18, 18))
    for i in range(3):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for ele in elements:
        ele_rot = ele.copy()
        if ele['local_z'] is not None:
            ele_rot['local_z'] = (R @ np.array(ele['local_z'])).tolist()
        elements_rot.append(ele_rot)
    u_global_rot = T @ u_global

    def mock_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        return np.eye(12) + np.random.rand(12, 12) * 0.01

    def mock_compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        return np.random.rand(12) * 0.5

    def mock_local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        np.random.seed(42)
        K_local = np.random.rand(12, 12) * 0.1
        return 0.5 * (K_local + K_local.T)
    with patch('__main__.beam_transformation_matrix_3D', mock_beam_transformation_matrix_3D), patch('__main__.compute_local_element_loads_beam_3D', mock_compute_local_element_loads_beam_3D), patch('__main__.local_geometric_stiffness_matrix_3D_beam', mock_local_geometric_stiffness_matrix_3D_beam):
        K_original = fcn(node_coords, elements, u_global)
        K_rotated = fcn(node_coords_rot, elements_rot, u_global_rot)
        K_expected = T @ K_original @ T.T
        assert np.allclose(K_rotated, K_expected, rtol=0.1, atol=0.001)