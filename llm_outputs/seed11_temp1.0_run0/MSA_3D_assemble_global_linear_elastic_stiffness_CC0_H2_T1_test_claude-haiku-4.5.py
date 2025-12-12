def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    node_coords_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_single = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 5e-05}]
    with patch('__main__.local_elastic_stiffness_matrix_3D_beam') as mock_local_stiffness, patch('__main__.beam_transformation_matrix_3D') as mock_transform:
        mock_local_stiffness.return_value = np.eye(12) * 1000000.0
        mock_transform.return_value = np.eye(12)
        K_single = fcn(node_coords_single, elements_single)
        assert K_single.shape == (12, 12), 'Single element should produce 12x12 matrix for 2 nodes'
        assert np.allclose(K_single, K_single.T, atol=1e-06), 'Single element matrix should be symmetric'
        assert np.any(K_single != 0), 'Single element matrix should have nonzero entries'
    node_coords_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_chain = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 5e-05}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 5e-05}]
    with patch('__main__.local_elastic_stiffness_matrix_3D_beam') as mock_local_stiffness, patch('__main__.beam_transformation_matrix_3D') as mock_transform:
        mock_local_stiffness.return_value = np.eye(12) * 1000000.0
        mock_transform.return_value = np.eye(12)
        K_chain = fcn(node_coords_chain, elements_chain)
        assert K_chain.shape == (18, 18), 'Linear chain should produce 18x18 matrix for 3 nodes'
        assert np.allclose(K_chain, K_chain.T, atol=1e-06), 'Linear chain matrix should be symmetric'
        assert np.any(K_chain != 0), 'Linear chain matrix should have nonzero entries'
        assert np.any(K_chain[6:12, 6:12] != 0), 'Middle node should have nonzero contribution'
    node_coords_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    elements_triangle = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 5e-05}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 5e-05}, {'node_i': 2, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 5e-05}]
    with patch('__main__.local_elastic_stiffness_matrix_3D_beam') as mock_local_stiffness, patch('__main__.beam_transformation_matrix_3D') as mock_transform:
        mock_local_stiffness.return_value = np.eye(12) * 1000000.0
        mock_transform.return_value = np.eye(12)
        K_triangle = fcn(node_coords_triangle, elements_triangle)
        assert K_triangle.shape == (18, 18), 'Triangle should produce 18x18 matrix for 3 nodes'
        assert np.allclose(K_triangle, K_triangle.T, atol=1e-06), 'Triangle matrix should be symmetric'
        assert np.any(K_triangle != 0), 'Triangle matrix should have nonzero entries'
    node_coords_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_square = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 5e-05}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 5e-05}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 5e-05}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 5e-05}]
    with patch('__main__.local_elastic_stiffness_matrix_3D_beam') as mock_local_stiffness, patch('__main__.beam_transformation_matrix_3D') as mock_transform:
        mock_local_stiffness.return_value = np.eye(12) * 1000000.0
        mock_transform.return_value = np.eye(12)
        K_square = fcn(node_coords_square, elements_square)
        assert K_square.shape == (24, 24), 'Square should produce 24x24 matrix for 4 nodes'
        assert np.allclose(K_square, K_square.T, atol=1e-06), 'Square matrix should be symmetric'
        assert np.any(K_square != 0), 'Square matrix should have nonzero entries'
        for i in range(4):
            block = K_square[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6]
            assert np.any(block != 0), f'Node {i} diagonal block should be nonzero'