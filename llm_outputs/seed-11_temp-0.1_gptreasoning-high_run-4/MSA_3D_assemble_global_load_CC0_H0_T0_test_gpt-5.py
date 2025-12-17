def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """

    def ref_assemble(nodal_loads, n_nodes):
        P = np.zeros(6 * n_nodes, dtype=float)
        for node, vec in nodal_loads.items():
            v = np.asarray(vec, dtype=float).reshape(6)
            start = 6 * node
            P[start:start + 6] = v
        return P
    P0 = fcn({}, 0)
    assert isinstance(P0, np.ndarray)
    assert P0.shape == (0,)
    assert P0.dtype.kind == 'f'
    loads1 = {0: [1, 2, 3, 4, 5, 6]}
    P1 = fcn(loads1, 1)
    exp1 = ref_assemble(loads1, 1)
    assert P1.shape == (6,)
    assert np.allclose(P1, exp1)
    assert P1.dtype.kind == 'f'
    n2 = 4
    loads2 = {0: [10, 0, -5, 0, 2, 0], 2: [0, 0, 0, -1, -2, -3], 3: [7, 8, 9, 10, 11, 12]}
    P2 = fcn(loads2, n2)
    exp2 = ref_assemble(loads2, n2)
    assert P2.shape == (6 * n2,)
    assert np.allclose(P2, exp2)
    assert np.allclose(P2[0:6], np.asarray(loads2[0], dtype=float))
    assert np.allclose(P2[12:18], np.asarray(loads2[2], dtype=float))
    assert np.allclose(P2[18:24], np.asarray(loads2[3], dtype=float))
    assert np.allclose(P2[6:12], 0.0)
    rng = np.random.default_rng(20231108)
    for _ in range(10):
        n = int(rng.integers(0, 15))
        nodal_loads = {}
        if n > 0:
            m = int(rng.integers(0, n + 1))
            if m > 0:
                nodes = rng.choice(np.arange(n), size=m, replace=False).tolist()
                for node in nodes:
                    if rng.random() < 0.5:
                        vec = rng.integers(-100, 101, size=6).tolist()
                    else:
                        vec = rng.normal(0, 100, size=6).tolist()
                    nodal_loads[int(node)] = vec
        P = fcn(nodal_loads, n)
        P_ref = ref_assemble(nodal_loads, n)
        assert P.shape == (6 * n,)
        assert P.dtype.kind == 'f'
        assert np.allclose(P, P_ref)
        all_nodes = set(range(n))
        specified = set(nodal_loads.keys())
        for u in sorted(all_nodes - specified):
            start = 6 * u
            assert np.allclose(P[start:start + 6], 0.0)
        for node, vec in nodal_loads.items():
            start = 6 * node
            assert np.allclose(P[start:start + 6], np.asarray(vec, dtype=float))