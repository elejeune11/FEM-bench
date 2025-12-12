def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    import numpy as np
    n_nodes = 1
    nodal_loads = {0: [1, 2, 3, 4, 5, 6]}
    P = np.asarray(fcn(nodal_loads, n_nodes))
    assert P.shape == (6 * n_nodes,)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    assert np.allclose(P, expected)
    assert P.dtype.kind == 'f'
    n_nodes = 5
    nodal_loads = {1: (10, -1, 0.5, 0, 0, 0), 3: np.array([0, 0, -2, 1, 2, 3])}
    P = np.asarray(fcn(nodal_loads, n_nodes))
    assert P.shape == (6 * n_nodes,)
    expected = np.zeros(6 * n_nodes, dtype=float)
    expected[6 * 1:6 * 1 + 6] = np.array([10.0, -1.0, 0.5, 0.0, 0.0, 0.0])
    expected[6 * 3:6 * 3 + 6] = np.array([0.0, 0.0, -2.0, 1.0, 2.0, 3.0])
    assert np.allclose(P, expected)
    node = 3
    idx = 6 * node
    assert P[idx + 0] == expected[idx + 0]
    assert P[idx + 1] == expected[idx + 1]
    assert P[idx + 2] == expected[idx + 2]
    assert P[idx + 3] == expected[idx + 3]
    assert P[idx + 4] == expected[idx + 4]
    assert P[idx + 5] == expected[idx + 5]
    rng = np.random.default_rng(12345)
    n_nodes = 12
    all_indices = np.arange(n_nodes)
    chosen = rng.choice(all_indices, size=6, replace=False)
    nodal_loads_random = {}
    expected_rand = np.zeros(6 * n_nodes, dtype=float)
    for (i, node) in enumerate(chosen):
        vec = rng.uniform(-100.0, 100.0, size=6)
        if i % 3 == 0:
            inp = vec.tolist()
        elif i % 3 == 1:
            inp = tuple(vec.tolist())
        else:
            inp = vec.copy()
        nodal_loads_random[int(node)] = inp
        expected_rand[6 * int(node):6 * int(node) + 6] = vec.astype(float)
    P_rand = np.asarray(fcn(nodal_loads_random, n_nodes))
    assert P_rand.shape == (6 * n_nodes,)
    assert np.allclose(P_rand, expected_rand)
    unspecified = set(all_indices) - set(chosen)
    for node in unspecified:
        slice_vals = P_rand[6 * int(node):6 * int(node) + 6]
        assert np.allclose(slice_vals, np.zeros(6, dtype=float))