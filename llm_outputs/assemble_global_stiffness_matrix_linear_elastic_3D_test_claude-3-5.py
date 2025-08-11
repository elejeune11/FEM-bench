def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    """
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    node_coords = np.array([[0, 0, 0], [3, 0, 0]])
    K = fcn(elements, node_coords)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert not np.allclose(K[0:6, 6:12], 0)
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    node_coords = np.array([[0, 0, 0], [3, 0, 0], [6, 0, 0]])
    K = fcn(elements, node_coords)
    assert K.shape == (18, 18)
    assert np.allclose(K, K.T)
    assert not np.allclose(K[6:12, 12:18], 0)
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    node_coords = np.array([[0, 0, 0], [3, 0, 0], [1.5, 2.6, 0]])
    K = fcn(elements, node_coords)
    assert K.shape == (18, 18)
    assert np.allclose(K, K.T)
    assert not np.allclose(K[0:6, 12:18], 0)
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    node_coords = np.array([[0, 0, 0], [3, 0, 0], [3, 3, 0], [0, 3, 0]])
    K = fcn(elements, node_coords)
    assert K.shape == (24, 24)
    assert np.allclose(K, K.T)
    assert not np.allclose(K[0:6, 18:24], 0)