def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    import numpy as np

    def ref_assemble(nodal_loads, n_nodes):
        P = np.zeros(6 * n_nodes, dtype=float)
        for node, vec in nodal_loads.items():
            assert 0 <= node < n_nodes
            v = np.asarray(vec, dtype=float)
            v = v.reshape(-1)
            assert v.shape == (6,)
            P[6 * node:6 * node + 6] = v
        return P
    n_nodes = 1
    nodal_loads = {0: [10.0, -5, 0, 1.5, 2.5, -3]}
    expected = ref_assemble(nodal_loads, n_nodes)
    result = np.asarray(fcn(nodal_loads, n_nodes), dtype=float).reshape(-1)
    assert result.shape == (6 * n_nodes,)
    assert np.allclose(result, expected)
    n_nodes = 4
    nodal_loads = {0: [1, 2, 3, 4, 5, 6], 2: [0, 0, -1, 0.1, -0.2, 0.3], 3: [-7, 8, -9, 10, -11, 12]}
    expected = ref_assemble(nodal_loads, n_nodes)
    result = np.asarray(fcn(nodal_loads, n_nodes), dtype=float).reshape(-1)
    assert result.shape == (6 * n_nodes,)
    assert np.allclose(result, expected)
    assert np.allclose(result[6 * 1:6 * 1 + 6], 0.0)
    n_nodes = 3
    nodal_loads = {}
    expected = ref_assemble(nodal_loads, n_nodes)
    result = np.asarray(fcn(nodal_loads, n_nodes), dtype=float).reshape(-1)
    assert result.shape == (6 * n_nodes,)
    assert np.allclose(result, expected)
    assert np.allclose(result, np.zeros(6 * n_nodes))
    n_nodes = 2
    nodal_loads = {1: [100, 200, 300, 400, 500, 600]}
    result = np.asarray(fcn(nodal_loads, n_nodes), dtype=float).reshape(-1)
    assert result.shape == (12,)
    assert np.allclose(result[0:6], 0.0)
    assert np.allclose(result[6:12], [100, 200, 300, 400, 500, 600])
    rng = np.random.default_rng(42)
    for _ in range(20):
        n_nodes = int(rng.integers(1, 12))
        k = int(rng.integers(0, n_nodes + 1))
        nodes = rng.choice(np.arange(n_nodes), size=k, replace=False).tolist()
        nodal_loads = {}
        for node in nodes:
            arr = rng.uniform(-1000.0, 1000.0, size=6)
            if rng.random() < 0.5:
                arr = arr.round().astype(int)
            choice = int(rng.integers(0, 3))
            if choice == 0:
                vec = arr.tolist()
            elif choice == 1:
                vec = tuple(arr.tolist())
            else:
                vec = np.array(arr)
            nodal_loads[int(node)] = vec
        expected = ref_assemble(nodal_loads, n_nodes)
        result = np.asarray(fcn(nodal_loads, n_nodes), dtype=float).reshape(-1)
        assert result.shape == (6 * n_nodes,)
        assert np.allclose(result, expected)