def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
and that each element contributes a nonzero 12x12 block to the appropriate location.
Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    n_nodes = len(node_coords)
    K = fcn(node_coords, elements)
    assert K.shape == (6 * n_nodes, 6 * n_nodes)
    assert np.allclose(K, K.T)
    assert np.any(K != 0)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    n_nodes = len(node_coords)
    K = fcn(node_coords, elements)
    assert K.shape == (6 * n_nodes, 6 * n_nodes)
    assert np.allclose(K, K.T)
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        sub_K = K[np.ix_(dofs, dofs)]
        assert np.any(sub_K != 0)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 2, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    n_nodes = len(node_coords)
    K = fcn(node_coords, elements)
    assert K.shape == (6 * n_nodes, 6 * n_nodes)
    assert np.allclose(K, K.T)
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        sub_K = K[np.ix_(dofs, dofs)]
        assert np.any(sub_K != 0)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    n_nodes = len(node_coords)
    K = fcn(node_coords, elements)
    assert K.shape == (6 * n_nodes, 6 * n_nodes)
    assert np.allclose(K, K.T)
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        dofs = np.r_[6 * i:6 * i + 6, 6 * j:6 * j + 6]
        sub_K = K[np.ix_(dofs, dofs)]
        assert np.any(sub_K != 0)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = []
    n_nodes = len(node_coords)
    K = fcn(node_coords, elements)
    assert K.shape == (6 * n_nodes, 6 * n_nodes)
    assert np.all(K == 0)
    node_coords = np.empty((0, 3))
    elements = []
    n_nodes = len(node_coords)
    K = fcn(node_coords, elements)
    assert K.shape == (0, 0)
    node_coords = np.array([[0.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': [0, 0, 1]}]
    n_nodes = len(node_coords)
    K = fcn(node_coords, elements)
    assert K.shape == (6 * n_nodes, 6 * n_nodes)
    assert np.allclose(K, K.T)
    assert np.any(K != 0)