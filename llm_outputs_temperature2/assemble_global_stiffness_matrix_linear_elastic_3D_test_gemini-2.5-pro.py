def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
and that each element contributes a nonzero 12x12 block to the appropriate location.
Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003}
    configurations = {'single_element': {'node_coords': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]), 'elements': [{'node_i': 0, 'node_j': 1, **elem_props}]}, 'linear_chain': {'node_coords': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]]), 'elements': [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}]}, 'triangle_loop': {'node_coords': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]]), 'elements': [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}, {'node_i': 2, 'node_j': 0, **elem_props}]}, 'square_loop': {'node_coords': np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]), 'elements': [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}, {'node_i': 2, 'node_j': 3, **elem_props}, {'node_i': 3, 'node_j': 0, **elem_props}]}}
    for (name, config) in configurations.items():
        node_coords = config['node_coords']
        elements = config['elements']
        n_nodes = node_coords.shape[0]
        K = fcn(node_coords, elements)
        expected_shape = (6 * n_nodes, 6 * n_nodes)
        assert K.shape == expected_shape, f"Failed shape test for '{name}' config"
        assert np.allclose(K, K.T), f"Failed symmetry test for '{name}' config"
        for element in elements:
            i = element['node_i']
            j = element['node_j']
            dof_i = slice(6 * i, 6 * (i + 1))
            dof_j = slice(6 * j, 6 * (j + 1))
            assert np.any(K[dof_i, dof_j]), f"Off-diagonal block K[{i},{j}] is all zero for '{name}'"
            assert np.any(K[dof_j, dof_i]), f"Off-diagonal block K[{j},{i}] is all zero for '{name}'"
        connected_nodes = {node for elem in elements for node in (elem['node_i'], elem['node_j'])}
        for node_idx in connected_nodes:
            dof_k = slice(6 * node_idx, 6 * (node_idx + 1))
            assert np.any(K[dof_k, dof_k]), f"Diagonal block K[{node_idx},{node_idx}] is all zero for '{name}'"