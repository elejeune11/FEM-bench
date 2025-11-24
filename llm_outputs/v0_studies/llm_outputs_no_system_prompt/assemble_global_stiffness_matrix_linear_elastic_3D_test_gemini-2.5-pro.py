def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
and that each element contributes a nonzero 12x12 block to the appropriate location.
Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003, 'local_z': [0, 0, 1]}
    single_element_nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    single_element_elems = [{'node_i': 0, 'node_j': 1, **elem_props}]
    chain_nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    chain_elems = [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}]
    triangle_nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, np.sqrt(3), 1.0]])
    triangle_elems = [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}, {'node_i': 2, 'node_j': 0, **elem_props}]
    square_nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    square_elems = [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}, {'node_i': 2, 'node_j': 3, **elem_props}, {'node_i': 3, 'node_j': 0, **elem_props}]
    configurations = [('single_element', single_element_nodes, single_element_elems), ('linear_chain', chain_nodes, chain_elems), ('triangle_loop', triangle_nodes, triangle_elems), ('square_loop', square_nodes, square_elems)]
    for (name, node_coords, elements) in configurations:
        n_nodes = node_coords.shape[0]
        dofs = 6 * n_nodes
        K = fcn(node_coords, elements)
        assert K.shape == (dofs, dofs), f'Failed shape test for {name}'
        assert np.allclose(K, K.T), f'Failed symmetry test for {name}'
        for elem in elements:
            i = elem['node_i']
            j = elem['node_j']
            s_i = slice(6 * i, 6 * (i + 1))
            s_j = slice(6 * j, 6 * (j + 1))
            assert np.sum(np.abs(K[s_i, s_i])) > 1e-09, f'Node {i} diagonal block is zero for {name}'
            assert np.sum(np.abs(K[s_j, s_j])) > 1e-09, f'Node {j} diagonal block is zero for {name}'
            assert np.sum(np.abs(K[s_i, s_j])) > 1e-09, f'Off-diagonal block ({i},{j}) is zero for {name}'
            assert np.sum(np.abs(K[s_j, s_i])) > 1e-09, f'Off-diagonal block ({j},{i}) is zero for {name}'