def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
and that each element contributes a nonzero 12x12 block to the appropriate location.
Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003, 'local_z': [0, 0, 1]}
    test_cases = {'single_element': {'node_coords': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), 'elements': [{'node_i': 0, 'node_j': 1, **elem_props}]}, 'linear_chain': {'node_coords': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]), 'elements': [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}]}, 'triangle_loop': {'node_coords': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]]), 'elements': [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}, {'node_i': 2, 'node_j': 0, **elem_props}]}, 'square_loop': {'node_coords': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), 'elements': [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}, {'node_i': 2, 'node_j': 3, **elem_props}, {'node_i': 3, 'node_j': 0, **elem_props}]}}
    for case in test_cases.values():
        node_coords = case['node_coords']
        elements = case['elements']
        n_nodes = node_coords.shape[0]
        expected_dofs = 6 * n_nodes
        K = fcn(node_coords, elements)
        assert K.shape == (expected_dofs, expected_dofs)
        assert np.allclose(K, K.T)
        for elem in elements:
            (i, j) = (elem['node_i'], elem['node_j'])
            indices = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
            element_submatrix = K[np.ix_(indices, indices)]
            assert np.any(element_submatrix != 0)
            assert np.all(np.diag(element_submatrix) >= 0)