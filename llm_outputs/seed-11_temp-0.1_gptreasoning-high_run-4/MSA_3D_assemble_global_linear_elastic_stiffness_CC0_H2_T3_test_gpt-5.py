def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block at the appropriate locations for multiple configurations:
    single element, linear chain, triangle loop, and square loop. Also verifies that off-diagonal 6x6 blocks between
    non-adjacent node pairs remain zero.
    """

    def blk(K, i, j):
        return K[6 * i:6 * i + 6, 6 * j:6 * j + 6]

    def element_dict(i, j, local_z=(0.0, 0.0, 1.0)):
        return {'node_i': i, 'node_j': j, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 2e-06, 'I_z': 3e-06, 'J': 1e-05, 'local_z': np.array(local_z, dtype=float)}
    tol_sym = 1e-08
    tol_zero = 1e-12
    nodes_single = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0]], dtype=float)
    elements_single = [element_dict(0, 1, (0.0, 0.0, 1.0))]
    K_single = fcn(nodes_single, elements_single)
    n_single = nodes_single.shape[0]
    assert K_single.shape == (6 * n_single, 6 * n_single)
    assert np.allclose(K_single, K_single.T, atol=tol_sym, rtol=0.0)
    idx01 = list(range(0, 6)) + list(range(6, 12))
    sub12_single = K_single[np.ix_(idx01, idx01)]
    assert np.linalg.norm(sub12_single, ord=np.inf) > 0.0
    assert np.linalg.norm(blk(K_single, 0, 1), ord=np.inf) > 0.0
    nodes_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    edges_chain = [(0, 1), (1, 2), (2, 3)]
    elements_chain = [element_dict(i, j, (0.0, 0.0, 1.0)) for i, j in edges_chain]
    K_chain = fcn(nodes_chain, elements_chain)
    n_chain = nodes_chain.shape[0]
    assert K_chain.shape == (6 * n_chain, 6 * n_chain)
    assert np.allclose(K_chain, K_chain.T, atol=tol_sym, rtol=0.0)
    for i, j in edges_chain:
        assert np.linalg.norm(blk(K_chain, i, j), ord=np.inf) > 0.0
        idx = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
        assert np.linalg.norm(K_chain[np.ix_(idx, idx)], ord=np.inf) > 0.0
    nonadj_chain = [(0, 2), (0, 3), (1, 3)]
    for i, j in nonadj_chain:
        assert np.allclose(blk(K_chain, i, j), 0.0, atol=tol_zero, rtol=0.0)
    nodes_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.4, 0.8, 0.0]], dtype=float)
    edges_triangle = [(0, 1), (1, 2), (2, 0)]
    elements_triangle = [element_dict(i, j, (0.0, 0.0, 1.0)) for i, j in edges_triangle]
    K_triangle = fcn(nodes_triangle, elements_triangle)
    n_triangle = nodes_triangle.shape[0]
    assert K_triangle.shape == (6 * n_triangle, 6 * n_triangle)
    assert np.allclose(K_triangle, K_triangle.T, atol=tol_sym, rtol=0.0)
    for i, j in edges_triangle:
        assert np.linalg.norm(blk(K_triangle, i, j), ord=np.inf) > 0.0
        idx = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
        assert np.linalg.norm(K_triangle[np.ix_(idx, idx)], ord=np.inf) > 0.0
    nodes_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    edges_square = [(0, 1), (1, 2), (2, 3), (3, 0)]
    elements_square = [element_dict(i, j, (0.0, 0.0, 1.0)) for i, j in edges_square]
    K_square = fcn(nodes_square, elements_square)
    n_square = nodes_square.shape[0]
    assert K_square.shape == (6 * n_square, 6 * n_square)
    assert np.allclose(K_square, K_square.T, atol=tol_sym, rtol=0.0)
    for i, j in edges_square:
        assert np.linalg.norm(blk(K_square, i, j), ord=np.inf) > 0.0
        idx = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
        assert np.linalg.norm(K_square[np.ix_(idx, idx)], ord=np.inf) > 0.0
    nonadj_square = [(0, 2), (1, 3)]
    for i, j in nonadj_square:
        assert np.allclose(blk(K_square, i, j), 0.0, atol=tol_zero, rtol=0.0)