def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    import numpy as np

    def ref_assemble(nodal_loads, n_nodes):
        P = np.zeros(6 * n_nodes, dtype=float)
        for (node, vec) in nodal_loads.items():
            P[6 * node:6 * node + 6] = np.asarray(vec, dtype=float)
        return P
    loads1 = {0: [1, 2, 3, 4, 5, 6]}
    n_nodes1 = 1
    expected1 = ref_assemble(loads1, n_nodes1)
    P1 = fcn(loads1, n_nodes1)
    assert isinstance(P1, np.ndarray)
    assert P1.shape == (6 * n_nodes1,)
    assert P1.dtype.kind == 'f'
    assert np.allclose(P1, expected1)
    n_nodes2 = 4
    loads2 = {0: [10, 0, -5, 0, 2, 3], 2: (7, 1, 0, -4, 0, 9), 3: np.array([0, 0, 0, 5, -6, 0], dtype=int)}
    expected2 = ref_assemble(loads2, n_nodes2)
    P2 = fcn(loads2, n_nodes2)
    assert isinstance(P2, np.ndarray)
    assert P2.shape == (6 * n_nodes2,)
    assert P2.dtype.kind == 'f'
    assert np.allclose(P2, expected2)
    assert np.allclose(P2[6 * 1:6 * 1 + 6], 0.0)
    assert np.allclose(P2[0:6], np.array(loads2[0], dtype=float))
    assert np.allclose(P2[12:18], np.array(loads2[2], dtype=float))
    assert np.allclose(P2[18:24], np.array(loads2[3], dtype=float))
    P0 = fcn({}, 3)
    assert isinstance(P0, np.ndarray)
    assert P0.shape == (18,)
    assert P0.dtype.kind == 'f'
    assert np.array_equal(P0, np.zeros(18, dtype=float))
    try:
        rng = np.random.default_rng(12345)
        use_generator = True
    except AttributeError:
        rng = np.random.RandomState(12345)
        use_generator = False

    def rand_int(low, high):
        if use_generator:
            return int(rng.integers(low, high))
        return int(rng.randint(low, high))
    for _ in range(10):
        n_nodes = rand_int(1, 15)
        node_indices = np.arange(n_nodes)
        k = rand_int(0, n_nodes + 1)
        if k == 0:
            chosen = np.array([], dtype=int)
        else:
            chosen = rng.choice(node_indices, size=k, replace=False)
        nodal_loads = {}
        for node in chosen:
            vec = rng.normal(loc=0.0, scale=10.0, size=6)
            if rand_int(0, 2) == 1:
                vec = np.rint(vec).astype(int)
            nodal_loads[int(node)] = vec
        expected = ref_assemble(nodal_loads, n_nodes)
        P = fcn(nodal_loads, n_nodes)
        assert isinstance(P, np.ndarray)
        assert P.shape == (6 * n_nodes,)
        assert P.dtype.kind == 'f'
        assert np.allclose(P, expected)