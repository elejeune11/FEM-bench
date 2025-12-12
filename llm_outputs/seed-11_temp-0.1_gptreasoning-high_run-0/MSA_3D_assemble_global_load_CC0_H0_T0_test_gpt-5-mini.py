def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    import numpy as np
    n_nodes = 1
    nodal_loads = {0: [1, 2, 3, 4, 5, 6]}
    P = fcn(nodal_loads, n_nodes)
    P_arr = np.asarray(P)
    assert P_arr.shape == (6 * n_nodes,)
    assert np.issubdtype(P_arr.dtype, np.floating)
    assert np.allclose(P_arr, np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float))
    n_nodes = 4
    nodal_loads = {1: (-10, 0.5, 1000.0, -2, -3, 0), 3: [0, 0, 0, 1, 2, 3]}
    P = fcn(nodal_loads, n_nodes)
    P_arr = np.asarray(P)
    assert P_arr.shape == (6 * n_nodes,)
    expected = np.zeros(6 * n_nodes, dtype=float)
    expected[6 * 1:6 * 1 + 6] = np.asarray(nodal_loads[1], dtype=float)
    expected[6 * 3:6 * 3 + 6] = np.asarray(nodal_loads[3], dtype=float)
    assert np.allclose(P_arr, expected)
    assert np.allclose(P_arr[0:6], np.zeros(6, dtype=float))
    assert np.allclose(P_arr[6:12], expected[6:12])
    assert np.allclose(P_arr[12:18], np.zeros(6, dtype=float))
    assert np.allclose(P_arr[18:24], expected[18:24])
    rng = np.random.default_rng(123456)
    n_nodes = 10
    n_loaded = 5
    loaded_nodes = tuple(rng.choice(n_nodes, size=n_loaded, replace=False))
    nodal_loads = {}
    expected = np.zeros(6 * n_nodes, dtype=float)
    loaded_mask = np.zeros(6 * n_nodes, dtype=bool)
    for node in loaded_nodes:
        node = int(node)
        vec = rng.normal(loc=0.0, scale=100.0, size=6).tolist()
        nodal_loads[node] = vec
        expected[6 * node:6 * node + 6] = np.asarray(vec, dtype=float)
        loaded_mask[6 * node:6 * node + 6] = True
    P = fcn(nodal_loads, n_nodes)
    P_arr = np.asarray(P)
    assert P_arr.shape == (6 * n_nodes,)
    assert np.allclose(P_arr, expected)
    assert np.issubdtype(P_arr.dtype, np.floating)
    assert np.allclose(P_arr[~loaded_mask], 0.0)