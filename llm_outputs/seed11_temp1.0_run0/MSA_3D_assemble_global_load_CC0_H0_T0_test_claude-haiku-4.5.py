def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    nodal_loads = {0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    n_nodes = 1
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (6,), 'Output shape should be (6,) for 1 node'
    assert np.allclose(P, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 'Single node load not correctly placed'
    nodal_loads = {0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2: [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]}
    n_nodes = 3
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (18,), 'Output shape should be (18,) for 3 nodes'
    assert np.allclose(P[0:6], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 'Node 0 load incorrectly placed'
    assert np.allclose(P[6:12], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), 'Node 1 (unspecified) should have zero loads'
    assert np.allclose(P[12:18], [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]), 'Node 2 load incorrectly placed'
    nodal_loads = {1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    n_nodes = 3
    P = fcn(nodal_loads, n_nodes)
    assert np.allclose(P[0:6], 0.0), 'Node 0 should be all zeros'
    assert np.allclose(P[6:12], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 'Node 1 load incorrectly placed'
    assert np.allclose(P[12:18], 0.0), 'Node 2 should be all zeros'
    nodal_loads = {0: [1.0, 1.0, 1.0, 1.0, 1.0, 1.0], 1: [2.0, 2.0, 2.0, 2.0, 2.0, 2.0], 2: [3.0, 3.0, 3.0, 3.0, 3.0, 3.0]}
    n_nodes = 3
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (18,)
    assert np.allclose(P[0:6], [1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
    assert np.allclose(P[6:12], [2.0, 2.0, 2.0, 2.0, 2.0, 2.0])
    assert np.allclose(P[12:18], [3.0, 3.0, 3.0, 3.0, 3.0, 3.0])
    nodal_loads = {}
    n_nodes = 2
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (12,)
    assert np.allclose(P, 0.0), 'Empty loads should result in all zeros'
    nodal_loads = {0: [1, 2, 3, 4, 5, 6]}
    n_nodes = 1
    P = fcn(nodal_loads, n_nodes)
    assert P.dtype in [np.float64, np.float32, float], 'Output should be float type'
    assert np.allclose(P, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    nodal_loads = {0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 99: [0.0, 0.0, 1.0, 0.0, 0.0, 0.0]}
    n_nodes = 100
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (600,)
    assert np.allclose(P[0:6], [1.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(P[594:600], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    assert np.allclose(P[6:594], 0.0)
    nodal_loads = {0: [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]}
    n_nodes = 1
    P = fcn(nodal_loads, n_nodes)
    assert np.allclose(P, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0])
    nodal_loads = {0: [1.0, -1.0, 2.0, -2.0, 3.0, -3.0], 1: [-1.0, 1.0, -2.0, 2.0, -3.0, 3.0]}
    n_nodes = 2
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (12,)
    assert np.allclose(P[0:6], [1.0, -1.0, 2.0, -2.0, 3.0, -3.0])
    assert np.allclose(P[6:12], [-1.0, 1.0, -2.0, 2.0, -3.0, 3.0])
    np.random.seed(42)
    n_nodes = 5
    nodal_loads = {}
    for i in range(n_nodes):
        if np.random.rand() > 0.5:
            nodal_loads[i] = list(np.random.randn(6))
    P = fcn(nodal_loads, n_nodes)
    P_expected = np.zeros(6 * n_nodes)
    for (node_idx, load) in nodal_loads.items():
        start_idx = 6 * node_idx
        P_expected[start_idx:start_idx + 6] = load
    assert np.allclose(P, P_expected), 'Function output does not match manual assembly'
    nodal_loads = {0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    n_nodes = 1
    P = fcn(nodal_loads, n_nodes)
    assert isinstance(P, np.ndarray), 'Output should be numpy ndarray'
    nodal_loads = {0: [1e-10, 10000000000.0, -1e-10, -10000000000.0, 1e-15, 1000000000000000.0]}
    n_nodes = 1
    P = fcn(nodal_loads, n_nodes)
    assert np.allclose(P, [1e-10, 10000000000.0, -1e-10, -10000000000.0, 1e-15, 1000000000000000.0])