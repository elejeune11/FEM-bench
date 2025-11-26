def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003}
    n_nodes_1 = 2
    node_coords_1 = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    elements_1 = [{'node_i': 0, 'node_j': 1, **elem_props}]
    K1 = fcn(node_coords_1, elements_1)
    assert K1.shape == (6 * n_nodes_1, 6 * n_nodes_1), 'Incorrect shape for single element case'
    assert np.allclose(K1, K1.T), 'Matrix is not symmetric for single element case'
    assert np.sum(np.abs(K1)) > 1e-09, 'Matrix is all zeros for single element case'
    n_nodes_2 = 3
    node_coords_2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    elements_2 = [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}]
    K2 = fcn(node_coords_2, elements_2)
    assert K2.shape == (6 * n_nodes_2, 6 * n_nodes_2), 'Incorrect shape for linear chain case'
    assert np.allclose(K2, K2.T), 'Matrix is not symmetric for linear chain case'
    for i in range(n_nodes_2):
        block = K2[6 * i:6 * (i + 1), 6 * i:6 * (i + 1)]
        assert np.sum(np.abs(block)) > 1e-09, f'Node {i} has no stiffness contribution in linear chain case'
    n_nodes_3 = 3
    node_coords_3 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, np.sqrt(3), 1.0]])
    elements_3 = [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}, {'node_i': 2, 'node_j': 0, **elem_props}]
    K3 = fcn(node_coords_3, elements_3)
    assert K3.shape == (6 * n_nodes_3, 6 * n_nodes_3), 'Incorrect shape for triangle loop case'
    assert np.allclose(K3, K3.T), 'Matrix is not symmetric for triangle loop case'
    for i in range(n_nodes_3):
        block = K3[6 * i:6 * (i + 1), 6 * i:6 * (i + 1)]
        assert np.sum(np.abs(block)) > 1e-09, f'Node {i} has no stiffness contribution in triangle loop case'
    n_nodes_4 = 4
    node_coords_4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_4 = [{'node_i': 0, 'node_j': 1, **elem_props}, {'node_i': 1, 'node_j': 2, **elem_props}, {'node_i': 2, 'node_j': 3, **elem_props}, {'node_i': 3, 'node_j': 0, **elem_props, 'local_z': [0, 0, 1]}]
    K4 = fcn(node_coords_4, elements_4)
    assert K4.shape == (6 * n_nodes_4, 6 * n_nodes_4), 'Incorrect shape for square loop case'
    assert np.allclose(K4, K4.T), 'Matrix is not symmetric for square loop case'
    for i in range(n_nodes_4):
        block = K4[6 * i:6 * (i + 1), 6 * i:6 * (i + 1)]
        assert np.sum(np.abs(block)) > 1e-09, f'Node {i} has no stiffness contribution in square loop case'