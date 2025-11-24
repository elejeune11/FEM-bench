@pytest.mark.parametrize('node_coords, elements', test_cases)
def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn, node_coords, elements):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    n_nodes = node_coords.shape[0]
    expected_shape = (6 * n_nodes, 6 * n_nodes)
    K = fcn(node_coords, elements)
    assert K.shape == expected_shape
    assert np.allclose(K, K.T)
    for element in elements:
        node_i = element['node_i']
        node_j = element['node_j']
        dofs = np.r_[6 * node_i:6 * node_i + 6, 6 * node_j:6 * node_j + 6]
        K_element_block = K[np.ix_(dofs, dofs)]
        assert np.any(K_element_block != 0)