def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    import numpy as np
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 0, 1]}]
    K = fcn(node_coords, elements)
    assert K.shape == (12, 12)
    assert np.allclose(K, K.T)
    assert np.linalg.norm(K) > 1e-10
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 0, 1]}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18)
    assert np.allclose(K, K.T)
    assert np.linalg.norm(K) > 1e-10
    assert np.linalg.norm(K[0:12, 0:12]) > 1e-10
    assert np.linalg.norm(K[6:18, 6:18]) > 1e-10
    assert np.linalg.norm(K[6:12, 6:12]) > 1e-10
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 0, 1]}, {'node_i': 2, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 0, 1]}]
    K = fcn(node_coords, elements)
    assert K.shape == (18, 18)
    assert np.allclose(K, K.T)
    assert np.linalg.norm(K) > 1e-10
    assert np.linalg.norm(K[0:12, 0:12]) > 1e-10
    assert np.linalg.norm(K[6:18, 6:18]) > 1e-10
    assert np.linalg.norm(K[12:18, 0:6]) > 1e-10
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 0, 1]}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 0, 1]}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 0, 1]}]
    K = fcn(node_coords, elements)
    assert K.shape == (24, 24)
    assert np.allclose(K, K.T)
    assert np.linalg.norm(K) > 1e-10
    assert np.linalg.norm(K[0:12, 0:12]) > 1e-10
    assert np.linalg.norm(K[6:18, 6:18]) > 1e-10
    assert np.linalg.norm(K[12:24, 12:24]) > 1e-10
    assert np.linalg.norm(K[18:24, 0:6]) > 1e-10
    assert np.linalg.norm(K[12:18, 18:24]) > 1e-10
    assert np.linalg.norm(K[0:6, 18:24]) > 1e-10