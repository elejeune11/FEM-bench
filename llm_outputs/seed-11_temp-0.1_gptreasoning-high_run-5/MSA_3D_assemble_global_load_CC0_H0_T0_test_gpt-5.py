def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    import numpy as np
    loads_single = {0: [1, 2, 3, 4, 5, 6]}
    P_single = fcn(loads_single, 1)
    assert isinstance(P_single, np.ndarray)
    assert P_single.shape == (6,)
    assert P_single.dtype.kind == 'f'
    assert np.allclose(P_single, np.array([1, 2, 3, 4, 5, 6], dtype=float))
    loads_multi = {0: [10, 0, 0, 0, 0, 0], 2: [0, 20, 0, 0, 0, 30]}
    n_multi = 4
    P_multi = fcn(loads_multi, n_multi)
    assert isinstance(P_multi, np.ndarray)
    assert P_multi.shape == (6 * n_multi,)
    assert P_multi.dtype.kind == 'f'
    assert np.allclose(P_multi[0:6], np.array(loads_multi[0], dtype=float))
    assert np.allclose(P_multi[6:12], np.zeros(6))
    assert np.allclose(P_multi[12:18], np.array(loads_multi[2], dtype=float))
    assert np.allclose(P_multi[18:24], np.zeros(6))
    loads_varied = {1: (7, 8, 9, 10, 11, 12), 3: np.array([13, 14, 15, 16, 17, 18], dtype=int)}
    n_varied = 5
    P_varied = fcn(loads_varied, n_varied)
    assert P_varied.shape == (6 * n_varied,)
    assert P_varied.dtype.kind == 'f'
    assert np.allclose(P_varied[0:6], 0)
    assert np.allclose(P_varied[6:12], np.array(loads_varied[1], dtype=float))
    assert np.allclose(P_varied[12:18], 0)
    assert np.allclose(P_varied[18:24], np.array(loads_varied[3], dtype=float))
    assert np.allclose(P_varied[24:30], 0)
    rng = np.random.default_rng(12345)
    for _ in range(20):
        n = int(rng.integers(1, 10))
        k = int(rng.integers(0, n + 1))
        chosen_nodes = list(rng.choice(np.arange(n), size=k, replace=False))
        nodal_loads_rand = {}
        P_ref = np.zeros(6 * n, dtype=float)
        for node in chosen_nodes:
            rep_choice = int(rng.integers(0, 3))
            vec_base = rng.normal(size=6)
            if rep_choice == 0:
                vec = vec_base.tolist()
            elif rep_choice == 1:
                vec = tuple(np.round(vec_base * 10).astype(int).tolist())
            elif rng.random() < 0.5:
                vec = np.array(vec_base, dtype=float)
            else:
                vec = np.array(np.round(vec_base * 100).astype(int))
            nodal_loads_rand[int(node)] = vec
            P_ref[6 * node:6 * node + 6] = np.array(vec, dtype=float)
        P_rand = fcn(nodal_loads_rand, n)
        assert isinstance(P_rand, np.ndarray)
        assert P_rand.shape == (6 * n,)
        assert P_rand.dtype.kind == 'f'
        assert np.allclose(P_rand, P_ref)
        unspecified_nodes = sorted(set(range(n)) - set(chosen_nodes))
        for node in unspecified_nodes:
            assert np.allclose(P_rand[6 * node:6 * node + 6], 0)
        shuffled_keys = list(nodal_loads_rand.keys())
        rng.shuffle(shuffled_keys)
        nodal_loads_shuffled = {int(k): nodal_loads_rand[k] for k in shuffled_keys}
        P_rand_shuffled = fcn(nodal_loads_shuffled, n)
        assert np.allclose(P_rand_shuffled, P_ref)