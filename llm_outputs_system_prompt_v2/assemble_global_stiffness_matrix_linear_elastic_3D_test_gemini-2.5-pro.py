def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    nodes1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements1 = [{'node_i': 0, 'node_j': 1, **elem_props}]
    nodes2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    elements2 = [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}]
    nodes3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    elements3 = [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}, {'node_i': 2, 'node_j': 0, **elem_props}]
    nodes4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements4 = [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}, {'node_i': 2, 'node_j': 3, **elem_props}, {'node_i': 3, 'node_j': 0, **elem_props}]
    test_cases = [(nodes1, elements1), (nodes2, elements2), (nodes3, elements3), (nodes4, elements4)]
    for (node_coords, elements) in test_cases:
        n_nodes = node_coords.shape[0]
        dof = 6 * n_nodes
        K = fcn(node_coords, elements)
        assert K.shape == (dof, dof)
        assert np.allclose(K, K.T)
        for elem in elements:
            (i, j) = (elem['node_i'], elem['node_j'])
            dofs_i = slice(6 * i, 6 * (i + 1))
            dofs_j = slice(6 * j, 6 * (j + 1))
            global_indices = np.r_[dofs_i, dofs_j]
            K_sub = K[np.ix_(global_indices, global_indices)]
            assert np.any(K_sub != 0)
            assert np.all(np.diag(K_sub) != 0)