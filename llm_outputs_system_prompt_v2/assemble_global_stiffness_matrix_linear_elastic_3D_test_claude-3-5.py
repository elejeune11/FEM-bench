def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location."""
    import numpy as np
    nodes_1 = np.array([[0, 0, 0], [1, 0, 0]])
    elements_1 = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    K1 = fcn(nodes_1, elements_1)
    assert K1.shape == (12, 12)
    assert np.allclose(K1, K1.T)
    assert not np.allclose(K1[0:6, 6:12], 0)
    nodes_2 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements_2 = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    K2 = fcn(nodes_2, elements_2)
    assert K2.shape == (18, 18)
    assert np.allclose(K2, K2.T)
    assert not np.allclose(K2[6:12, 12:18], 0)
    nodes_3 = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(0.75), 0]])
    elements_3 = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    K3 = fcn(nodes_3, elements_3)
    assert K3.shape == (18, 18)
    assert np.allclose(K3, K3.T)
    nodes_4 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements_4 = [{'node_i': i, 'node_j': (i + 1) % 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002} for i in range(4)]
    K4 = fcn(nodes_4, elements_4)
    assert K4.shape == (24, 24)
    assert np.allclose(K4, K4.T)
    eigenvals = np.linalg.eigvals(K4)
    assert np.sum(eigenvals > -1e-10) == len(eigenvals)