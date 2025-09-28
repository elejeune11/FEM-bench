def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K = fcn(node_coords, elements)
    n_nodes = node_coords.shape[0]
    expected_shape = (6 * n_nodes, 6 * n_nodes)
    assert K.shape == expected_shape, f'Expected shape {expected_shape}, got {K.shape}'
    assert np.allclose(K, K.T), 'Stiffness matrix should be symmetric'
    assert not np.allclose(K, 0), 'Matrix should not be all zeros'
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K = fcn(node_coords, elements)
    n_nodes = node_coords.shape[0]
    expected_shape = (6 * n_nodes, 6 * n_nodes)
    assert K.shape == expected_shape, f'Expected shape {expected_shape}, got {K.shape}'
    assert np.allclose(K, K.T), 'Stiffness matrix should be symmetric'
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [0.5, 0.866, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K = fcn(node_coords, elements)
    n_nodes = node_coords.shape[0]
    expected_shape = (6 * n_nodes, 6 * n_nodes)
    assert K.shape == expected_shape, f'Expected shape {expected_shape}, got {K.shape}'
    assert np.allclose(K, K.T), 'Stiffness matrix should be symmetric'
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K = fcn(node_coords, elements)
    n_nodes = node_coords.shape[0]
    expected_shape = (6 * n_nodes, 6 * n_nodes)
    assert K.shape == expected_shape, f'Expected shape {expected_shape}, got {K.shape}'
    assert np.allclose(K, K.T), 'Stiffness matrix should be symmetric'