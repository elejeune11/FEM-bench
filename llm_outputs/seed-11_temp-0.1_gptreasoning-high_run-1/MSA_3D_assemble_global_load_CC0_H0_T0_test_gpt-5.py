def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    loads = {0: [1, 2, 3, 4, 5, 6]}
    n_nodes = 1
    P = np.asarray(fcn(loads, n_nodes))
    assert P.shape == (6,)
    assert P.ndim == 1
    assert np.issubdtype(P.dtype, np.floating)
    expected = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    assert np.allclose(P, expected)
    loads = {0: [10, 0, -5, 2, 0, 1], 2: [0, 3, 0, 0, 0.5, 0]}
    n_nodes = 3
    P = np.asarray(fcn(loads, n_nodes))
    expected = np.concatenate([np.array([10, 0, -5, 2, 0, 1], dtype=float), np.zeros(6, dtype=float), np.array([0, 3, 0, 0, 0.5, 0], dtype=float)])
    assert P.shape == (6 * n_nodes,)
    assert np.allclose(P, expected)
    assert np.allclose(P[6:12], 0.0)
    assert np.allclose(P[12:18], np.array([0, 3, 0, 0, 0.5, 0], dtype=float))
    rng = np.random.default_rng(12345)
    for _ in range(20):
        n_nodes = int(rng.integers(1, 11))
        k = int(rng.integers(0, n_nodes + 1))
        nodes = sorted(rng.choice(n_nodes, size=k, replace=False).tolist())
        nodal_loads = {}
        for idx in nodes:
            choice = int(rng.integers(0, 3))
            if choice == 0:
                v = rng.normal(size=6).tolist()
            elif choice == 1:
                v = tuple(rng.random(6) * 200 - 100)
            else:
                v = rng.integers(-1000, 1000, size=6)
            nodal_loads[int(idx)] = v
        P_ref = np.zeros(6 * n_nodes, dtype=float)
        for idx, v in nodal_loads.items():
            dv = np.asarray(v, dtype=float).reshape(6)
            start = 6 * idx
            P_ref[start:start + 6] = dv
        P = np.asarray(fcn(nodal_loads, n_nodes))
        assert P.shape == P_ref.shape
        assert P.ndim == 1
        assert np.issubdtype(P.dtype, np.floating)
        assert np.allclose(P, P_ref)
        unspecified = set(range(n_nodes)) - set(nodal_loads.keys())
        if unspecified:
            some_idx = next(iter(unspecified))
            sl = slice(6 * some_idx, 6 * some_idx + 6)
            assert np.allclose(P[sl], 0.0)
    n_nodes = 4
    nodal_loads = {1: [1, 2, 3, 4, 5, 6], 3: [7, 8, 9, 10, 11, 12]}
    P = np.asarray(fcn(nodal_loads, n_nodes))
    assert np.allclose(P[6:12], np.array([1, 2, 3, 4, 5, 6], dtype=float))
    assert np.allclose(P[18:24], np.array([7, 8, 9, 10, 11, 12], dtype=float))
    assert np.allclose(P[0:6], 0.0)
    assert np.allclose(P[12:18], 0.0)