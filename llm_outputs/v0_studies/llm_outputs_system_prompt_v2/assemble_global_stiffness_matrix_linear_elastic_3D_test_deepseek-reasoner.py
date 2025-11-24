def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    import numpy as np
    import pytest

    def local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J):
        k = np.eye(12)
        return k

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        T = np.eye(12)
        return T

    def create_single_element():
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 0.0001}]
        return (node_coords, elements)

    def create_linear_chain():
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 0.0001}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 0.0001}]
        return (node_coords, elements)

    def create_triangle_loop():
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]])
        elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 0.0001}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 0.0001}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 0.0001}]
        return (node_coords, elements)

    def create_square_loop():
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 0.0001}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 0.0001}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 0.0001}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 0.0001}]
        return (node_coords, elements)
    test_configs = [('single_element', create_single_element()), ('linear_chain', create_linear_chain()), ('triangle_loop', create_triangle_loop()), ('square_loop', create_square_loop())]
    for (config_name, (node_coords, elements)) in test_configs:
        K = fcn(node_coords, elements)
        n_nodes = node_coords.shape[0]
        expected_shape = (6 * n_nodes, 6 * n_nodes)
        assert K.shape == expected_shape, f'Shape mismatch for {config_name}'
        assert np.allclose(K, K.T), f'Matrix not symmetric for {config_name}'
        if config_name == 'single_element':
            assert not np.allclose(K, 0), 'Single element matrix should be non-zero'
            assert K[0:12, 0:12].any(), '12x12 block should be non-zero for single element'