def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}]
    K = fcn(node_coords, elements)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert np.any(K[:6, :6] != 0)
    assert np.any(K[:6, 6:] != 0)
    assert np.any(K[6:, :6] != 0)
    assert np.any(K[6:, 6:] != 0)
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18)
    assert np.allclose(K, K.T)
    assert np.any(K[:6, :6] != 0)
    assert np.any(K[:6, 6:12] != 0)
    assert np.any(K[6:12, :6] != 0)
    assert np.any(K[6:12, 6:12] != 0)
    assert np.any(K[6:12, 12:] != 0)
    assert np.any(K[12:, 6:12] != 0)
    assert np.any(K[12:, 12:] != 0)
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18)
    assert np.allclose(K, K.T)
    for i in range(3):
        for j in range(3):
            block_i = slice(i * 6, (i + 1) * 6)
            block_j = slice(j * 6, (j + 1) * 6)
            assert np.any(K[block_i, block_j] != 0)
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}]
    K = fcn(node_coords, elements)
    assert K.shape == (24, 24)
    assert np.allclose(K, K.T)
    connected_pairs = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for (i, j) in connected_pairs:
        block_i = slice(i * 6, (i + 1) * 6)
        block_j = slice(j * 6, (j + 1) * 6)
        assert np.any(K[block_i, block_j] != 0)
        assert np.any(K[block_j, block_i] != 0)
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 1, 1]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': [0, 0, 1]}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': [1, 0, 0]}]
    K = fcn(node_coords, elements)
    assert K.shape == (24, 24)
    assert np.allclose(K, K.T)
    assert not np.any(np.isnan(K))
    assert not np.any(np.isinf(K))