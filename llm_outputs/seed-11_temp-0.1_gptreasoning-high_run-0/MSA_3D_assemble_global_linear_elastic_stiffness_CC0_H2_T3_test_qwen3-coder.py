def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}]
    K = fcn(node_coords, elements)
    n_dofs = 6 * node_coords.shape[0]
    assert K.shape == (n_dofs, n_dofs), 'Stiffness matrix shape is incorrect for single element.'
    assert np.allclose(K, K.T), 'Stiffness matrix is not symmetric for single element.'
    assert not np.allclose(K, 0), 'Stiffness matrix is zero for single element.'
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}]
    K = fcn(node_coords, elements)
    n_dofs = 6 * node_coords.shape[0]
    assert K.shape == (n_dofs, n_dofs), 'Stiffness matrix shape is incorrect for linear chain.'
    assert np.allclose(K, K.T), 'Stiffness matrix is not symmetric for linear chain.'
    assert not np.allclose(K, 0), 'Stiffness matrix is zero for linear chain.'
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}]
    K = fcn(node_coords, elements)
    n_dofs = 6 * node_coords.shape[0]
    assert K.shape == (n_dofs, n_dofs), 'Stiffness matrix shape is incorrect for triangle loop.'
    assert np.allclose(K, K.T), 'Stiffness matrix is not symmetric for triangle loop.'
    assert not np.allclose(K, 0), 'Stiffness matrix is zero for triangle loop.'
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}]
    K = fcn(node_coords, elements)
    n_dofs = 6 * node_coords.shape[0]
    assert K.shape == (n_dofs, n_dofs), 'Stiffness matrix shape is incorrect for square loop.'
    assert np.allclose(K, K.T), 'Stiffness matrix is not symmetric for square loop.'
    assert not np.allclose(K, 0), 'Stiffness matrix is zero for square loop.'