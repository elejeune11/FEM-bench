def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """
    import numpy as np

    def as_dense(K):
        return K.toarray() if hasattr(K, 'toarray') else np.array(K, dtype=float, copy=False)

    def run_case(node_coords, elements, expected_connected_pairs, expected_zero_pairs):
        node_coords = np.array(node_coords, dtype=float)
        K = as_dense(fcn(node_coords, elements))
        n = len(node_coords)
        assert K.shape == (6 * n, 6 * n)
        assert np.allclose(K, K.T, atol=1e-09, rtol=1e-09)
        assert np.linalg.norm(K) > 0
        for (i, j) in expected_connected_pairs:
            idx = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
            sub = K[np.ix_(idx, idx)]
            assert np.linalg.norm(sub) > 0
            sub_ii = K[6 * i:6 * i + 6, 6 * i:6 * i + 6]
            sub_jj = K[6 * j:6 * j + 6, 6 * j:6 * j + 6]
            sub_ij = K[6 * i:6 * i + 6, 6 * j:6 * j + 6]
            sub_ji = K[6 * j:6 * j + 6, 6 * i:6 * i + 6]
            assert np.linalg.norm(sub_ii) > 0
            assert np.linalg.norm(sub_jj) > 0
            assert np.linalg.norm(sub_ij) > 0
            assert np.allclose(sub_ij, sub_ji.T, atol=1e-09, rtol=1e-09)
        for (i, j) in expected_zero_pairs:
            sub_ij = K[6 * i:6 * i + 6, 6 * j:6 * j + 6]
            sub_ji = K[6 * j:6 * j + 6, 6 * i:6 * i + 6]
            assert np.allclose(sub_ij, 0.0, atol=1e-14)
            assert np.allclose(sub_ji, 0.0, atol=1e-14)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I = 1e-06
    J = 2e-06
    local_z = [0.0, 0.0, 1.0]
    nodes1 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    elements1 = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z}]
    run_case(nodes1, elements1, expected_connected_pairs=[(0, 1)], expected_zero_pairs=[])
    nodes2 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    elements2 = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z}]
    run_case(nodes2, elements2, expected_connected_pairs=[(0, 1), (1, 2)], expected_zero_pairs=[(0, 2)])
    s3 = np.sqrt(3.0) / 2.0
    nodes3 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, s3, 0.0)]
    elements3 = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z}, {'node_i': 2, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z}]
    run_case(nodes3, elements3, expected_connected_pairs=[(0, 1), (1, 2), (2, 0)], expected_zero_pairs=[])
    nodes4 = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    elements4 = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I, 'I_z': I, 'J': J, 'local_z': local_z}]
    run_case(nodes4, elements4, expected_connected_pairs=[(0, 1), (1, 2), (2, 3), (3, 0)], expected_zero_pairs=[(0, 2), (1, 3)])