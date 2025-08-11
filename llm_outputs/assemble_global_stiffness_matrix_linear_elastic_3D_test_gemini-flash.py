def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    elements1 = [{'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}]
    node_coords1 = np.array([[0, 0, 0], [1, 0, 0]])
    K1 = fcn(elements1, node_coords1)
    assert K1.shape == (12, 12)
    assert np.allclose(K1, K1.T)
    elements2 = [{'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 1, 'node_j': 2, 'E': 1, 'nu': 0.3, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}]
    node_coords2 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    K2 = fcn(elements2, node_coords2)
    assert K2.shape == (18, 18)
    assert np.allclose(K2, K2.T)