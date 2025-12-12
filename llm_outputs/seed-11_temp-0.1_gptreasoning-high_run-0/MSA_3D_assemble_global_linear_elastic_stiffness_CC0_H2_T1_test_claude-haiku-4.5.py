def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    node_coords_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_single = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}]
    K_single = fcn(node_coords_single, elements_single)
    assert K_single.shape == (12, 12), 'Single element should produce 12x12 matrix'
    assert_array_almost_equal(K_single, K_single.T, decimal=10, err_msg='Stiffness matrix should be symmetric for single element')
    assert np.linalg.norm(K_single) > 0, 'Stiffness matrix should be nonzero'
    node_coords_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_chain = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}]
    K_chain = fcn(node_coords_chain, elements_chain)
    assert K_chain.shape == (18, 18), 'Linear chain of 3 nodes should produce 18x18 matrix'
    assert_array_almost_equal(K_chain, K_chain.T, decimal=10, err_msg='Stiffness matrix should be symmetric for linear chain')
    assert np.linalg.norm(K_chain) > 0, 'Stiffness matrix should be nonzero'
    K_interior_block = K_chain[6:12, 6:12]
    assert np.linalg.norm(K_interior_block) > 0, 'Interior node should have nonzero stiffness contributions'
    node_coords_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    elements_triangle = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}, {'node_i': 2, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}]
    K_triangle = fcn(node_coords_triangle, elements_triangle)
    assert K_triangle.shape == (18, 18), 'Triangle loop should produce 18x18 matrix'
    assert_array_almost_equal(K_triangle, K_triangle.T, decimal=10, err_msg='Stiffness matrix should be symmetric for triangle loop')
    assert np.linalg.norm(K_triangle) > 0, 'Stiffness matrix should be nonzero'
    node_coords_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_square = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-05, 'I_z': 8.333e-05, 'J': 0.0001667}]
    K_square = fcn(node_coords_square, elements_square)
    assert K_square.shape == (24, 24), 'Square loop should produce 24x24 matrix'
    assert_array_almost_equal(K_square, K_square.T, decimal=10, err_msg='Stiffness matrix should be symmetric for square loop')
    assert np.linalg.norm(K_square) > 0, 'Stiffness matrix should be nonzero'
    for node_idx in range(4):
        node_block = K_square[node_idx * 6:(node_idx + 1) * 6, node_idx * 6:(node_idx + 1) * 6]
        assert np.linalg.norm(node_block) > 0, f'Node {node_idx} should have nonzero stiffness contributions'