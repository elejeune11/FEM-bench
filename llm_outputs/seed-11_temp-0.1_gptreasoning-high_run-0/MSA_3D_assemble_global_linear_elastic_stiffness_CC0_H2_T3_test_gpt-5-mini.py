def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    import numpy as np
    tol = 1e-12

    def _check(node_coords, elements):
        node_coords = np.asarray(node_coords, dtype=float)
        K = fcn(node_coords, elements)
        if hasattr(K, 'toarray'):
            K = K.toarray()
        else:
            K = np.asarray(K, dtype=float)
        n = node_coords.shape[0]
        assert K.shape == (6 * n, 6 * n)
        assert np.allclose(K, K.T, atol=1e-08, rtol=1e-06)
        for e in elements:
            i = int(e['node_i'])
            j = int(e['node_j'])
            rows = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
            block_12 = K[np.ix_(rows, rows)]
            assert np.linalg.norm(block_12, ord='fro') > tol
            assert np.allclose(block_12, block_12.T, atol=1e-08, rtol=1e-06)
            ii = list(range(6 * i, 6 * i + 6))
            jj = list(range(6 * j, 6 * j + 6))
            block_ii = K[np.ix_(ii, ii)]
            block_jj = K[np.ix_(jj, jj)]
            block_ij = K[np.ix_(ii, jj)]
            block_ji = K[np.ix_(jj, ii)]
            assert np.linalg.norm(block_ii, ord='fro') > tol
            assert np.linalg.norm(block_jj, ord='fro') > tol
            assert np.linalg.norm(block_ij, ord='fro') > tol
            assert np.linalg.norm(block_ji, ord='fro') > tol
    base_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06}
    nodes1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    el1 = dict(node_i=0, node_j=1, **base_props)
    _check(nodes1, [el1])
    nodes2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    el2_1 = dict(node_i=0, node_j=1, **base_props)
    el2_2 = dict(node_i=1, node_j=2, **base_props)
    _check(nodes2, [el2_1, el2_2])
    tri_h = np.sqrt(3.0) / 2.0
    nodes3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, tri_h, 0.0]], dtype=float)
    el3_1 = dict(node_i=0, node_j=1, **base_props)
    el3_2 = dict(node_i=1, node_j=2, **base_props)
    el3_3 = dict(node_i=2, node_j=0, **base_props)
    _check(nodes3, [el3_1, el3_2, el3_3])
    nodes4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    el4_1 = dict(node_i=0, node_j=1, **base_props)
    el4_2 = dict(node_i=1, node_j=2, **base_props)
    el4_3 = dict(node_i=2, node_j=3, **base_props)
    el4_4 = dict(node_i=3, node_j=0, **base_props)
    _check(nodes4, [el4_1, el4_2, el4_3, el4_4])