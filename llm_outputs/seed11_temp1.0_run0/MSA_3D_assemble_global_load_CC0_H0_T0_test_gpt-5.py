def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    loads_single = {0: [1, 2, 3, 4, 5, 6]}
    P_single = fcn(loads_single, 1)
    expected_single = np.array([1, 2, 3, 4, 5, 6], dtype=float)
    assert isinstance(P_single, np.ndarray)
    assert P_single.shape == (6,)
    assert np.issubdtype(P_single.dtype, np.floating)
    np.testing.assert_allclose(P_single, expected_single, rtol=0, atol=1e-12)
    P_zero = fcn({}, 3)
    expected_zero = np.zeros(6 * 3, dtype=float)
    assert P_zero.shape == (18,)
    np.testing.assert_allclose(P_zero, expected_zero, rtol=0, atol=1e-12)
    n_nodes_multi = 4
    loads_multi = {0: [10, 0, -5, 1.5, 0, 0], 2: (0.0, 2.2, 0.0, 0.0, -7.0, 8.0), 3: np.array([0, 0, 0, 9, 9, 9], dtype=int)}
    P_multi = fcn(loads_multi, n_nodes_multi)
    expected_multi = np.zeros(6 * n_nodes_multi, dtype=float)
    expected_multi[0:6] = np.array([10, 0, -5, 1.5, 0, 0], dtype=float)
    expected_multi[12:18] = np.array([0.0, 2.2, 0.0, 0.0, -7.0, 8.0], dtype=float)
    expected_multi[18:24] = np.array([0.0, 0.0, 0.0, 9.0, 9.0, 9.0], dtype=float)
    np.testing.assert_allclose(P_multi[6:12], np.zeros(6), rtol=0, atol=1e-12)
    np.testing.assert_allclose(P_multi, expected_multi, rtol=0, atol=1e-12)
    rng = np.random.RandomState(42)
    n_nodes_rand = 10
    chosen = rng.choice(n_nodes_rand, size=6, replace=False)
    loads_rand = {}
    for idx in chosen:
        vec = rng.randn(6)
        if idx % 3 == 0:
            loads_rand[idx] = list(vec)
        elif idx % 3 == 1:
            loads_rand[idx] = tuple(vec)
        else:
            loads_rand[idx] = (vec * 10).astype(int)
    P_rand = fcn(loads_rand, n_nodes_rand)
    expected_rand = np.zeros(6 * n_nodes_rand, dtype=float)
    for (node, vec) in loads_rand.items():
        vec_arr = np.array(vec, dtype=float)
        start = 6 * node
        expected_rand[start:start + 6] = vec_arr
    np.testing.assert_allclose(P_rand, expected_rand, rtol=0, atol=1e-12)
    not_loaded = set(range(n_nodes_rand)).difference(loads_rand.keys())
    for node in not_loaded:
        seg = P_rand[6 * node:6 * node + 6]
        np.testing.assert_allclose(seg, np.zeros(6), rtol=0, atol=1e-12)