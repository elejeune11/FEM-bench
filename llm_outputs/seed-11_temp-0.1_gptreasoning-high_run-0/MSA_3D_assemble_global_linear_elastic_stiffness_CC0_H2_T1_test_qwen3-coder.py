def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """

    def local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J):
        return np.ones((12, 12))

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        return np.eye(12)
    fcn.__globals__.update({'local_elastic_stiffness_matrix_3D_beam': local_elastic_stiffness_matrix_3D_beam, 'beam_transformation_matrix_3D': beam_transformation_matrix_3D})
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}]
    K = fcn(node_coords, elements)
    assert K.shape == (12, 12), 'Stiffness matrix shape is incorrect for single element.'
    assert np.allclose(K, K.T), 'Stiffness matrix is not symmetric for single element.'
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18), 'Stiffness matrix shape is incorrect for linear chain.'
    assert np.allclose(K, K.T), 'Stiffness matrix is not symmetric for linear chain.'
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [0.5, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18), 'Stiffness matrix shape is incorrect for triangle loop.'
    assert np.allclose(K, K.T), 'Stiffness matrix is not symmetric for triangle loop.'
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}]
    K = fcn(node_coords, elements)
    assert K.shape == (24, 24), 'Stiffness matrix shape is incorrect for square loop.'
    assert np.allclose(K, K.T), 'Stiffness matrix is not symmetric for square loop.'
    for elem in elements:
        (i, j) = (elem['node_i'], elem['node_j'])
        rows = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
        cols = rows
        block = K[np.ix_(rows, cols)]
        assert not np.allclose(block, 0), f'Element from node {i} to {j} did not contribute a nonzero block.'