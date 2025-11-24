def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    node_coords_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_single = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K_single = fcn(node_coords_single, elements_single)
    n_nodes = len(node_coords_single)
    assert K_single.shape == (6 * n_nodes, 6 * n_nodes)
    assert np.allclose(K_single, K_single.T)
    element_block = K_single[0:12, 0:12]
    assert not np.allclose(element_block, np.zeros((12, 12)))
    node_coords_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_chain = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K_chain = fcn(node_coords_chain, elements_chain)
    n_nodes_chain = len(node_coords_chain)
    assert K_chain.shape == (6 * n_nodes_chain, 6 * n_nodes_chain)
    assert np.allclose(K_chain, K_chain.T)
    block1 = K_chain[0:12, 0:12]
    assert not np.allclose(block1, np.zeros((12, 12)))
    block2 = K_chain[6:18, 6:18]
    assert not np.allclose(block2, np.zeros((12, 12)))
    node_coords_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]])
    elements_triangle = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K_triangle = fcn(node_coords_triangle, elements_triangle)
    n_nodes_triangle = len(node_coords_triangle)
    assert K_triangle.shape == (6 * n_nodes_triangle, 6 * n_nodes_triangle)
    assert np.allclose(K_triangle, K_triangle.T)
    node_coords_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_square = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-06, 'I_z': 8.33e-06, 'J': 1.67e-06}]
    K_square = fcn(node_coords_square, elements_square)
    n_nodes_square = len(node_coords_square)
    assert K_square.shape == (6 * n_nodes_square, 6 * n_nodes_square)
    assert np.allclose(K_square, K_square.T)