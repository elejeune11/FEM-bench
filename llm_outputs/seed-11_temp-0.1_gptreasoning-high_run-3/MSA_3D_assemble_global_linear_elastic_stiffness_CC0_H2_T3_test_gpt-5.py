def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """
    import numpy as np
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-06
    I_z = 1.2e-06
    J = 2e-06
    local_z = [0.0, 0.0, 1.0]

    def mk_elem(i, j):
        return {'node_i': int(i), 'node_j': int(j), 'E': float(E), 'nu': float(nu), 'A': float(A), 'I_y': float(I_y), 'I_z': float(I_z), 'J': float(J), 'local_z': np.array(local_z, dtype=float)}

    def run_case(node_coords, elements, connected_pairs, disconnected_pairs):
        node_coords = np.array(node_coords, dtype=float)
        K = fcn(node_coords, elements)
        n_nodes = node_coords.shape[0]
        assert K.shape == (6 * n_nodes, 6 * n_nodes)
        assert np.allclose(K, K.T, atol=1e-08, rtol=1e-08)
        for i, j in connected_pairs:
            si = slice(6 * i, 6 * (i + 1))
            sj = slice(6 * j, 6 * (j + 1))
            K_ii = K[si, si]
            K_jj = K[sj, sj]
            K_ij = K[si, sj]
            K_ji = K[sj, si]
            assert np.linalg.norm(K_ii) > 0.0
            assert np.linalg.norm(K_jj) > 0.0
            assert np.linalg.norm(K_ij) > 0.0
            assert np.allclose(K_ij, K_ji.T, atol=1e-08, rtol=1e-08)
        for i, j in disconnected_pairs:
            si = slice(6 * i, 6 * (i + 1))
            sj = slice(6 * j, 6 * (j + 1))
            K_ij = K[si, sj]
            maxK = np.max(np.abs(K)) if K.size else 0.0
            if maxK == 0.0:
                assert np.allclose(K_ij, 0.0)
            else:
                assert np.max(np.abs(K_ij)) <= maxK * 1e-12
    nodes_single = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    elems_single = [mk_elem(0, 1)]
    run_case(nodes_single, elems_single, connected_pairs=[(0, 1)], disconnected_pairs=[])
    nodes_chain = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    elems_chain = [mk_elem(0, 1), mk_elem(1, 2)]
    run_case(nodes_chain, elems_chain, connected_pairs=[(0, 1), (1, 2)], disconnected_pairs=[(0, 2)])
    K_chain = fcn(np.array(nodes_chain, dtype=float), elems_chain)
    K00 = K_chain[0:6, 0:6]
    K11 = K_chain[6:12, 6:12]
    K22 = K_chain[12:18, 12:18]
    n00 = np.linalg.norm(K00)
    n11 = np.linalg.norm(K11)
    n22 = np.linalg.norm(K22)
    assert n11 > max(n00, n22)
    nodes_triangle = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0]]
    elems_triangle = [mk_elem(0, 1), mk_elem(1, 2), mk_elem(2, 0)]
    run_case(nodes_triangle, elems_triangle, connected_pairs=[(0, 1), (1, 2), (2, 0)], disconnected_pairs=[])
    nodes_square = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    elems_square = [mk_elem(0, 1), mk_elem(1, 2), mk_elem(2, 3), mk_elem(3, 0)]
    run_case(nodes_square, elems_square, connected_pairs=[(0, 1), (1, 2), (2, 3), (3, 0)], disconnected_pairs=[(0, 2), (1, 3)])