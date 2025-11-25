def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests shape, symmetry, and block structure of assembled global stiffness matrix."""
    import numpy as np
    nodes_1 = np.array([[0, 0, 0], [1, 0, 0]])
    elements_1 = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}]
    K1 = fcn(nodes_1, elements_1)
    assert K1.shape == (12, 12)
    assert np.allclose(K1, K1.T)
    assert np.count_nonzero(K1) > 0
    nodes_2 = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements_2 = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}]
    K2 = fcn(nodes_2, elements_2)
    assert K2.shape == (18, 18)
    assert np.allclose(K2, K2.T)
    nodes_3 = np.array([[0, 0, 0], [1, 0, 0], [0.5, np.sqrt(0.75), 0]])
    elements_3 = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}]
    K3 = fcn(nodes_3, elements_3)
    assert K3.shape == (18, 18)
    assert np.allclose(K3, K3.T)
    nodes_4 = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0], [0, 1, 0]])
    elements_4 = [{'node_i': i, 'node_j': (i + 1) % 4, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06} for i in range(4)]
    K4 = fcn(nodes_4, elements_4)
    assert K4.shape == (24, 24)
    assert np.allclose(K4, K4.T)
    for K in [K1, K2, K3, K4]:
        n = K.shape[0] // 6
        for i in range(n):
            for j in range(n):
                block = K[6 * i:6 * (i + 1), 6 * j:6 * (j + 1)]
                if i == j:
                    assert np.count_nonzero(block) > 0