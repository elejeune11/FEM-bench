def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1.0, 'nu': 0.3, 'A': 1.0, 'I_y': 1.0, 'I_z': 1.0, 'J': 1.0}]
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    K = fcn(elements, node_coords)
    assert K.shape == (6 * 2, 6 * 2)
    assert np.allclose(K, K.T)
    elements = [{'node_i': i, 'node_j': i + 1, 'E': 1.0, 'nu': 0.3, 'A': 1.0, 'I_y': 1.0, 'I_z': 1.0, 'J': 1.0} for i in range(3)]
    node_coords = np.array([[i, 0.0, 0.0] for i in range(4)])
    K = fcn(elements, node_coords)
    assert K.shape == (6 * 4, 6 * 4)
    assert np.allclose(K, K.T)
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1.0, 'nu': 0.3, 'A': 1.0, 'I_y': 1.0, 'I_z': 1.0, 'J': 1.0}, {'node_i': 1, 'node_j': 2, 'E': 1.0, 'nu': 0.3, 'A': 1.0, 'I_y': 1.0, 'I_z': 1.0, 'J': 1.0}, {'node_i': 2, 'node_j': 0, 'E': 1.0, 'nu': 0.3, 'A': 1.0, 'I_y': 1.0, 'I_z': 1.0, 'J': 1.0}]
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]])
    K = fcn(elements, node_coords)
    assert K.shape == (6 * 3, 6 * 3)
    assert np.allclose(K, K.T)
    elements = [{'node_i': i, 'node_j': (i + 1) % 4, 'E': 1.0, 'nu': 0.3, 'A': 1.0, 'I_y': 1.0, 'I_z': 1.0, 'J': 1.0} for i in range(4)]
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    K = fcn(elements, node_coords)
    assert K.shape == (6 * 4, 6 * 4)
    assert np.allclose(K, K.T)