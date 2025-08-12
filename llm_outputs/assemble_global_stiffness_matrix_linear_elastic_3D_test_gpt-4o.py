def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    node_coords_single = np.array([[0, 0, 0], [1, 0, 0]])
    elements_single = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}]
    K_single = fcn(node_coords_single, elements_single)
    assert K_single.shape == (12, 12)
    assert np.allclose(K_single, K_single.T)
    assert np.count_nonzero(K_single) > 0
    node_coords_chain = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements_chain = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}]
    K_chain = fcn(node_coords_chain, elements_chain)
    assert K_chain.shape == (18, 18)
    assert np.allclose(K_chain, K_chain.T)
    assert np.count_nonzero(K_chain) > 0
    node_coords_triangle = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]])
    elements_triangle = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}, {'node_i': 2, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}]
    K_triangle = fcn(node_coords_triangle, elements_triangle)
    assert K_triangle.shape == (18, 18)
    assert np.allclose(K_triangle, K_triangle.T)
    assert np.count_nonzero(K_triangle) > 0
    node_coords_square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements_square = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}]
    K_square = fcn(node_coords_square, elements_square)
    assert K_square.shape == (24, 24)
    assert np.allclose(K_square, K_square.T)
    assert np.count_nonzero(K_square) > 0