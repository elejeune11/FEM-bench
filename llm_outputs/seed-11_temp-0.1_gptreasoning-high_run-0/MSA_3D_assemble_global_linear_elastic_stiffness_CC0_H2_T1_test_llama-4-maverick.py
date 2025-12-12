def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1000000.0, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}]
    K = fcn(node_coords, elements)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1000000.0, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 1, 'node_j': 2, 'E': 1000000.0, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18)
    assert np.allclose(K, K.T)
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1000000.0, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 1, 'node_j': 2, 'E': 1000000.0, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 2, 'node_j': 3, 'E': 1000000.0, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 3, 'node_j': 0, 'E': 1000000.0, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}]
    K = fcn(node_coords, elements)
    assert K.shape == (24, 24)
    assert np.allclose(K, K.T)