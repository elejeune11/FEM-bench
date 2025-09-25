def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    element_props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05}
    node_coords_1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_1 = [{'node_i': 0, 'node_j': 1, **element_props}]
    K_1 = fcn(node_coords_1, elements_1)
    n_nodes_1 = node_coords_1.shape[0]
    expected_dim_1 = 6 * n_nodes_1
    assert K_1.shape == (expected_dim_1, expected_dim_1), f'Single element: Expected shape {expected_dim_1}x{expected_dim_1}, got {K_1.shape}'
    assert np.allclose(K_1, K_1.T, atol=1e-09), 'Single element: Global stiffness matrix is not symmetric'
    assert not np.allclose(K_1, np.zeros_like(K_1), atol=1e-09), 'Single element: Global stiffness matrix is all zeros'
    (node_i_1, node_j_1) = (elements_1[0]['node_i'], elements_1[0]['node_j'])
    dofs_i_1 = slice(6 * node_i_1, 6 * node_i_1 + 6)
    dofs_j_1 = slice(6 * node_j_1, 6 * node_j_1 + 6)
    element_dofs_1 = np.concatenate((np.arange(dofs_i_1.start, dofs_i_1.stop), np.arange(dofs_j_1.start, dofs_j_1.stop)))
    element_block_1 = K_1[np.ix_(element_dofs_1, element_dofs_1)]
    assert not np.allclose(element_block_1, np.zeros_like(element_block_1), atol=1e-09), "Single element: Element's contribution block is all zeros"
    node_coords_2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_2 = [{'node_i': 0, 'node_j': 1, **element_props}, {'node_i': 1, 'node_j': 2, **element_props}]
    K_2 = fcn(node_coords_2, elements_2)
    n_nodes_2 = node_coords_2.shape[0]
    expected_dim_2 = 6 * n_nodes_2
    assert K_2.shape == (expected_dim_2, expected_dim_2), f'Linear chain: Expected shape {expected_dim_2}x{expected_dim_2}, got {K_2.shape}'
    assert np.allclose(K_2, K_2.T, atol=1e-09), 'Linear chain: Global stiffness matrix is not symmetric'
    assert not np.allclose(K_2, np.zeros_like(K_2), atol=1e-09), 'Linear chain: Global stiffness matrix is all zeros'
    for (i, element) in enumerate(elements_2):
        (node_i, node_j) = (element['node_i'], element['node_j'])
        dofs_i = slice(6 * node_i, 6 * node_i + 6)
        dofs_j = slice(6 * node_j, 6 * node_j + 6)
        element_dofs = np.concatenate((np.arange(dofs_i.start, dofs_i.stop), np.arange(dofs_j.start, dofs_j.stop)))
        element_block = K_2[np.ix_(element_dofs, element_dofs)]
        assert not np.allclose(element_block, np.zeros_like(element_block), atol=1e-09), f"Linear chain: Element {i}'s contribution block is all zeros"
    node_coords_3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(0.75), 0.0]])
    elements_3 = [{'node_i': 0, 'node_j': 1, **element_props}, {'node_i': 1, 'node_j': 2, **element_props}, {'node_i': 2, 'node_j': 0, **element_props}]
    K_3 = fcn(node_coords_3, elements_3)
    n_nodes_3 = node_coords_3.shape[0]
    expected_dim_3 = 6 * n_nodes_3
    assert K_3.shape == (expected_dim_3, expected_dim_3), f'Triangle loop: Expected shape {expected_dim_3}x{expected_dim_3}, got {K_3.shape}'
    assert np.allclose(K_3, K_3.T, atol=1e-09), 'Triangle loop: Global stiffness matrix is not symmetric'
    assert not np.allclose(K_3, np.zeros_like(K_3), atol=1e-09), 'Triangle loop: Global stiffness matrix is all zeros'
    for (i, element) in enumerate(elements_3):
        (node_i, node_j) = (element['node_i'], element['node_j'])
        dofs_i = slice(6 * node_i, 6 * node_i + 6)
        dofs_j = slice(6 * node_j, 6 * node_j + 6)
        element_dofs = np.concatenate((np.arange(dofs_i.start, dofs_i.stop), np.arange(dofs_j.start, dofs_j.stop)))
        element_block = K_3[np.ix_(element_dofs, element_dofs)]
        assert not np.allclose(element_block, np.zeros_like(element_block), atol=1e-09), f"Triangle loop: Element {i}'s contribution block is all zeros"
    node_coords_4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_4 = [{'node_i': 0, 'node_j': 1, **element_props}, {'node_i': 1, 'node_j': 2, **element_props}, {'node_i': 2, 'node_j': 3, **element_props}, {'node_i': 3, 'node_j': 0, **element_props}]
    K_4 = fcn(node_coords_4, elements_4)
    n_nodes_4 = node_coords_4.shape[0]
    expected_dim_4 = 6 * n_nodes_4
    assert K_4.shape == (expected_dim_4, expected_dim_4), f'Square loop: Expected shape {expected_dim_4}x{expected_dim_4}, got {K_4.shape}'
    assert np.allclose(K_4, K_4.T, atol=1e-09), 'Square loop: Global stiffness matrix is not symmetric'
    assert not np.allclose(K_4, np.zeros_like(K_4), atol=1e-09), 'Square loop: Global stiffness matrix is all zeros'
    for (i, element) in enumerate(elements_4):
        (node_i, node_j) = (element['node_i'], element['node_j'])
        dofs_i = slice(6 * node_i, 6 * node_i + 6)
        dofs_j = slice(6 * node_j, 6 * node_j + 6)
        element_dofs = np.concatenate((np.arange(dofs_i.start, dofs_i.stop), np.arange(dofs_j.start, dofs_j.stop)))
        element_block = K_4[np.ix_(element_dofs, element_dofs)]
        assert not np.allclose(element_block, np.zeros_like(element_block), atol=1e-09), f"Square loop: Element {i}'s contribution block is all zeros"