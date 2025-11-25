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
    n_nodes = 3
    n_dof = 6 * n_nodes

    def mock_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        return np.eye(12)

    def mock_compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        return np.array([1.0, 0.5, 0.0, 0.2, 0.3, 0.1, -1.0, -0.5, 0.0, -0.2, -0.3, -0.1])

    def mock_local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        k_g = np.random.rand(12, 12)
        k_g = 0.5 * (k_g + k_g.T)
        return k_g * 0.1
    with patch('__main__.beam_transformation_matrix_3D', mock_beam_transformation_matrix_3D), patch('__main__.compute_local_element_loads_beam_3D', mock_compute_local_element_loads_beam_3D), patch('__main__.local_geometric_stiffness_matrix_3D_beam', mock_local_geometric_stiffness_matrix_3D_beam):
        u_zero = np.zeros(n_dof)
        with patch('__main__.compute_local_element_loads_beam_3D', return_value=np.zeros(12)):
            with patch('__main__.local_geometric_stiffness_matrix_3D_beam', return_value=np.zeros((12, 12))):
                K_zero = fcn(node_coords, elements, u_zero)
                assert np.allclose(K_zero, 0.0), 'Zero displacement should produce zero matrix'
        u_test = np.random.rand(n_dof) * 0.1
        K_test = fcn(node_coords, elements, u_test)
        assert np.allclose(K_test, K_test.T), 'Assembled matrix should be symmetric'
        scale_factor = 2.5
        u_scaled = scale_factor * u_test
        K_scaled = fcn(node_coords, elements, u_scaled)
        K_expected = scale_factor * K_test
        assert np.allclose(K_scaled, K_expected, rtol=1e-10), 'Scaling displacement should scale K_g linearly'
        u1 = np.random.rand(n_dof) * 0.1
        u2 = np.random.rand(n_dof) * 0.1
        K1 = fcn(node_coords, elements, u1)
        K2 = fcn(node_coords, elements, u2)
        K_sum = fcn(node_coords, elements, u1 + u2)
        assert np.allclose(K_sum, K1 + K2, rtol=1e-10), 'Superposition should hold'
        elements_reordered = [elements[1], elements[0]]
        K_reordered = fcn(node_coords, elements_reordered, u_test)
        assert np.allclose(K_test, K_reordered), 'Element order should not affect assembled result'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': [0, 0, 1]}]
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    node_coords_rot = (R @ node_coords.T).T
    local_z_rot = R @ np.array([0, 0, 1])
    elements_rot = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': local_z_rot}]
    n_nodes = 2
    n_dof = 6 * n_nodes
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        idx = i * 6
        T[idx:idx + 3, idx:idx + 3] = R
        T[idx + 3:idx + 6, idx + 3:idx + 6] = R
    u_global = np.random.rand(n_dof) * 0.1
    u_global_rot = T @ u_global

    def mock_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        return np.eye(12)

    def mock_compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        disp_mag = np.linalg.norm(u_e_global)
        return np.array([disp_mag, 0.5 * disp_mag, 0.0, 0.2 * disp_mag, 0.3 * disp_mag, 0.1 * disp_mag, -disp_mag, -0.5 * disp_mag, 0.0, -0.2 * disp_mag, -0.3 * disp_mag, -0.1 * disp_mag])

    def mock_local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        k_g = np.ones((12, 12)) * (abs(Fx2) + abs(Mx2) + abs(My1) + abs(Mz1) + abs(My2) + abs(Mz2)) * 0.01
        k_g = 0.5 * (k_g + k_g.T)
        return k_g
    with patch('__main__.beam_transformation_matrix_3D', mock_beam_transformation_matrix_3D), patch('__main__.compute_local_element_loads_beam_3D', mock_compute_local_element_loads_beam_3D), patch('__main__.local_geometric_stiffness_matrix_3D_beam', mock_local_geometric_stiffness_matrix_3D_beam):
        K_g = fcn(node_coords, elements, u_global)
        K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
        K_g_expected = T @ K_g @ T.T
        assert np.allclose(K_g_rot, K_g_expected, rtol=1e-10, atol=1e-12), 'Frame objectivity violated: K_g_rot should equal T @ K_g @ T.T'