def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.66e-05}]
    K = fcn(node_coords, elements)
    assert K.shape == (12, 12), 'Single element: incorrect shape'
    assert np.allclose(K, K.T), 'Single element: matrix not symmetric'
    assert np.linalg.norm(K) > 0, 'Single element: matrix is zero'
    assert np.linalg.norm(K[0:6, 0:6]) > 0, 'Single element: node 0-0 block is zero'
    assert np.linalg.norm(K[0:6, 6:12]) > 0, 'Single element: node 0-1 block is zero'
    assert np.linalg.norm(K[6:12, 0:6]) > 0, 'Single element: node 1-0 block is zero'
    assert np.linalg.norm(K[6:12, 6:12]) > 0, 'Single element: node 1-1 block is zero'
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.66e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.66e-05}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18), 'Linear chain: incorrect shape'
    assert np.allclose(K, K.T), 'Linear chain: matrix not symmetric'
    assert np.linalg.norm(K) > 0, 'Linear chain: matrix is zero'
    assert np.linalg.norm(K[6:12, 6:12]) > np.linalg.norm(K[0:6, 0:6]), 'Linear chain: middle node should have larger stiffness'
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.66e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.66e-05}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.66e-05}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18), 'Triangle loop: incorrect shape'
    assert np.allclose(K, K.T), 'Triangle loop: matrix not symmetric'
    assert np.linalg.norm(K) > 0, 'Triangle loop: matrix is zero'
    diag_norm_0 = np.linalg.norm(K[0:6, 0:6])
    diag_norm_1 = np.linalg.norm(K[6:12, 6:12])
    diag_norm_2 = np.linalg.norm(K[12:18, 12:18])
    assert np.allclose(diag_norm_0, diag_norm_1, rtol=0.1), 'Triangle loop: diagonal blocks should be similar'
    assert np.allclose(diag_norm_1, diag_norm_2, rtol=0.1), 'Triangle loop: diagonal blocks should be similar'
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.66e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.66e-05}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.66e-05}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.66e-05}]
    K = fcn(node_coords, elements)
    assert K.shape == (24, 24), 'Square loop: incorrect shape'
    assert np.allclose(K, K.T), 'Square loop: matrix not symmetric'
    assert np.linalg.norm(K) > 0, 'Square loop: matrix is zero'
    diag_norms = [np.linalg.norm(K[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6]) for i in range(4)]
    for i in range(1, 4):
        assert np.allclose(diag_norms[0], diag_norms[i], rtol=0.1), f'Square loop: corner {i} diagonal block differs from corner 0'
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.66e-05, 'local_z': [0, 0, 1]}]
    K = fcn(node_coords, elements)
    assert K.shape == (12, 12), 'With local_z: incorrect shape'
    assert np.allclose(K, K.T), 'With local_z: matrix not symmetric'
    assert np.linalg.norm(K) > 0, 'With local_z: matrix is zero'