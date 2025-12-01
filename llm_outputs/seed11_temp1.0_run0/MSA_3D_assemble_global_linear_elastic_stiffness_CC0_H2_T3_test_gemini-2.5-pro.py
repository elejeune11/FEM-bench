def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
and that each element contributes a nonzero 12x12 block to the appropriate location.
Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003}
    node_coords_1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_1 = [{'node_i': 0, 'node_j': 1, **props}]
    node_coords_2 = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0], [2.0, 1.0, 1.0]])
    elements_2 = [{'node_i': 0, 'node_j': 1, **props}, {'node_i': 1, 'node_j': 2, **props}]
    node_coords_3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    elements_3 = [{'node_i': 0, 'node_j': 1, **props}, {'node_i': 1, 'node_j': 2, **props}, {'node_i': 2, 'node_j': 0, **props}]
    node_coords_4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_4 = [{'node_i': 0, 'node_j': 1, **props}, {'node_i': 1, 'node_j': 2, **props}, {'node_i': 2, 'node_j': 3, **props}, {'node_i': 3, 'node_j': 0, **props}]
    node_coords_5 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    elements_5 = [{'node_i': 0, 'node_j': 1, **props}, {'node_i': 1, 'node_j': 2, 'local_z': [0, 0, 1], **props}]
    configurations = [('single_element', node_coords_1, elements_1), ('linear_chain', node_coords_2, elements_2), ('triangle_loop', node_coords_3, elements_3), ('square_loop', node_coords_4, elements_4), ('3d_structure', node_coords_5, elements_5)]
    for (name, node_coords, elements) in configurations:
        n_nodes = node_coords.shape[0]
        K = fcn(node_coords, elements)
        expected_shape = (6 * n_nodes, 6 * n_nodes)
        assert K.shape == expected_shape, f'Failed shape check for {name}'
        assert np.allclose(K, K.T), f'Failed symmetry check for {name}'
        for (i, element) in enumerate(elements):
            node_i = element['node_i']
            node_j = element['node_j']
            dofs_i = np.arange(6 * node_i, 6 * node_i + 6)
            dofs_j = np.arange(6 * node_j, 6 * node_j + 6)
            dofs = np.union1d(dofs_i, dofs_j)
            K_element_block = K[np.ix_(dofs, dofs)]
            assert np.any(K_element_block != 0), f'Element {i} in {name} has zero contribution'