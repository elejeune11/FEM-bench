def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    node_coords_single = np.array([[0, 0, 0], [1, 0, 0]])
    elements_single = [{'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}]
    K_single = fcn(node_coords_single, elements_single)
    assert K_single.shape == (12, 12)
    assert np.allclose(K_single, K_single.T)
    node_coords_chain = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements_chain = [{'node_i': i, 'node_j': i + 1, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1} for i in range(2)]
    K_chain = fcn(node_coords_chain, elements_chain)
    assert K_chain.shape == (18, 18)
    assert np.allclose(K_chain, K_chain.T)
    node_coords_triangle = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(3) / 2, 0]])
    elements_triangle = [{'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 1, 'node_j': 2, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 2, 'node_j': 0, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}]
    K_triangle = fcn(node_coords_triangle, elements_triangle)
    assert K_triangle.shape == (18, 18)
    assert np.allclose(K_triangle, K_triangle.T)
    node_coords_square = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements_square = [{'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 1, 'node_j': 2, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 2, 'node_j': 3, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 3, 'node_j': 0, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}]
    K_square = fcn(node_coords_square, elements_square)
    assert K_square.shape == (24, 24)
    assert np.allclose(K_square, K_square.T)