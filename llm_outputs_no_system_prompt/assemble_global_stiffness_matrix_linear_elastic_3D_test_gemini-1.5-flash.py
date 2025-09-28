def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 1, 'node_j': 2, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 2, 'node_j': 3, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}, {'node_i': 3, 'node_j': 0, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}]
    K = fcn(node_coords, elements)
    n_nodes = len(node_coords)
    assert K.shape == (6 * n_nodes, 6 * n_nodes)
    assert np.allclose(K, K.T)
    for element in elements:
        (i, j) = (element['node_i'], element['node_j'])
        rows = np.concatenate([np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)])
        cols = rows
        assert np.any(K[np.ix_(rows, cols)])
    node_coords_single = np.array([[0, 0, 0], [1, 0, 0]])
    elements_single = [{'node_i': 0, 'node_j': 1, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1}]
    K_single = fcn(node_coords_single, elements_single)
    assert K_single.shape == (12, 12)
    assert np.allclose(K_single, K_single.T)
    node_coords_chain = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements_chain = [{'node_i': i, 'node_j': i + 1, 'E': 1, 'nu': 0, 'A': 1, 'I_y': 1, 'I_z': 1, 'J': 1} for i in range(2)]
    K_chain = fcn(node_coords_chain, elements_chain)
    assert K_chain.shape == (18, 18)
    assert np.allclose(K_chain, K_chain.T)