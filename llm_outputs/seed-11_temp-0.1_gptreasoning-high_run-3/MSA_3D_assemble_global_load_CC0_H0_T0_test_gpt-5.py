def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    rng = np.random.default_rng(42)

    def manual_assemble(loads, n_nodes):
        P = np.zeros(6 * n_nodes, dtype=float)
        for node, vec in loads.items():
            start = 6 * node
            P[start:start + 6] = np.asarray(vec, dtype=float).reshape(6)
        return P
    loads = {0: [1, 2, 3, 4, 5, 6]}
    n_nodes = 1
    P = fcn(loads, n_nodes)
    assert isinstance(P, np.ndarray)
    assert P.shape == (6 * n_nodes,)
    assert P.dtype.kind == 'f'
    assert np.allclose(P, np.array([1, 2, 3, 4, 5, 6], dtype=float))
    n_nodes = 4
    loads = {0: (1, 0, 0, 0, 0, 0), 2: np.array([0, 2, 0, 0, 0, 0]), 3: [0, 0, 0, 3, 4, 5]}
    P = fcn(loads, n_nodes)
    assert isinstance(P, np.ndarray)
    assert P.shape == (6 * n_nodes,)
    assert P.dtype.kind == 'f'
    expected = np.zeros(6 * n_nodes, dtype=float)
    expected[0:6] = np.array([1, 0, 0, 0, 0, 0], dtype=float)
    expected[12:18] = np.array([0, 2, 0, 0, 0, 0], dtype=float)
    expected[18:24] = np.array([0, 0, 0, 3, 4, 5], dtype=float)
    assert np.allclose(P, expected)
    assert np.allclose(P[6:12], 0.0)
    n_nodes = 3
    loads = {0: [0, 5.0, 0, 0, 0, 0], 1: [0, 0, 0, 0, 0, 7.0], 2: [0, 0, 0, 0, 9.0, 0]}
    P = fcn(loads, n_nodes)
    assert P.shape == (6 * n_nodes,)
    assert P.dtype.kind == 'f'
    expected = np.zeros(6 * n_nodes, dtype=float)
    expected[1] = 5.0
    expected[11] = 7.0
    expected[16] = 9.0
    assert np.allclose(P, expected)
    n_nodes = 5
    loads = {}
    P = fcn(loads, n_nodes)
    assert P.shape == (6 * n_nodes,)
    assert P.dtype.kind == 'f'
    assert np.allclose(P, 0.0)
    for n_nodes in [1, 2, 5, 10]:
        for _ in range(5):
            node_indices = rng.choice(n_nodes, size=rng.integers(0, n_nodes + 1), replace=False)
            loads = {}
            for node in node_indices:
                vec = rng.normal(size=6).astype(float)
                kind = rng.integers(0, 3)
                if kind == 0:
                    loads[node] = vec.tolist()
                elif kind == 1:
                    loads[node] = tuple(vec)
                else:
                    loads[node] = np.array(vec)
            P_ref = manual_assemble(loads, n_nodes)
            P = fcn(loads, n_nodes)
            assert isinstance(P, np.ndarray)
            assert P.shape == (6 * n_nodes,)
            assert P.dtype.kind == 'f'
            assert np.allclose(P, P_ref, rtol=1e-12, atol=1e-12)