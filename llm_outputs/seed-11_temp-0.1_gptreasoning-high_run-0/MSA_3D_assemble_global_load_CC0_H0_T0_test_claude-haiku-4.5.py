def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    nodal_loads = {0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    P = fcn(nodal_loads, 1)
    assert P.shape == (6,), 'Output shape should be (6,) for 1 node'
    assert P[0] == 1.0, 'First DOF should be 1.0'
    assert np.allclose(P[1:], 0.0), 'Other DOFs should be zero'
    nodal_loads = {0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    P = fcn(nodal_loads, 1)
    assert P.shape == (6,), 'Output shape should be (6,) for 1 node'
    assert np.allclose(P, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 'All load components should match'
    nodal_loads = {0: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}
    n_nodes = 3
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (18,), 'Output shape should be (18,) for 3 nodes'
    assert np.allclose(P[0:6], [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]), 'Node 0 loads should match'
    assert np.allclose(P[6:12], 0.0), 'Node 1 should have zero loads'
    assert np.allclose(P[12:18], 0.0), 'Node 2 should have zero loads'
    nodal_loads = {1: [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]}
    n_nodes = 3
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (18,), 'Output shape should be (18,) for 3 nodes'
    assert np.allclose(P[0:6], 0.0), 'Node 0 should have zero loads'
    assert np.allclose(P[6:12], [5.0, 10.0, 15.0, 20.0, 25.0, 30.0]), 'Node 1 loads should match'
    assert np.allclose(P[12:18], 0.0), 'Node 2 should have zero loads'
    nodal_loads = {2: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]}
    n_nodes = 3
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (18,), 'Output shape should be (18,) for 3 nodes'
    assert np.allclose(P[0:6], 0.0), 'Node 0 should have zero loads'
    assert np.allclose(P[6:12], 0.0), 'Node 1 should have zero loads'
    assert np.allclose(P[12:18], [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]), 'Node 2 loads should match'
    nodal_loads = {0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 2: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}
    n_nodes = 3
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (18,), 'Output shape should be (18,) for 3 nodes'
    assert np.allclose(P[0:6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 'Node 0 loads should match'
    assert np.allclose(P[6:12], 0.0), 'Node 1 should have zero loads'
    assert np.allclose(P[12:18], [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]), 'Node 2 loads should match'
    nodal_loads = {}
    n_nodes = 2
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (12,), 'Output shape should be (12,) for 2 nodes'
    assert np.allclose(P, 0.0), 'All loads should be zero for empty input'
    nodal_loads = {0: [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]}
    P = fcn(nodal_loads, 1)
    assert P.dtype in [np.float32, np.float64], 'Output should be float type'
    assert np.allclose(P, [1.5, 2.5, 3.5, 4.5, 5.5, 6.5]), 'Float values should be preserved'
    nodal_loads = {0: [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]}
    P = fcn(nodal_loads, 1)
    assert np.allclose(P, [-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]), 'Negative loads should be preserved'
    nodal_loads = {0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 99: [0.0, 1.0, 0.0, 0.0, 0.0, 0.0]}
    n_nodes = 100
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (600,), 'Output shape should be (600,) for 100 nodes'
    assert P[0] == 1.0, 'Node 0, DOF 0 should be 1.0'
    assert P[6 * 99 + 1] == 1.0, 'Node 99, DOF 1 should be 1.0'
    np.random.seed(42)
    n_nodes = 5
    nodal_loads = {}
    expected_P = np.zeros(6 * n_nodes)
    for node_idx in range(n_nodes):
        if np.random.rand() > 0.5:
            load_vector = np.random.randn(6) * 100
            nodal_loads[node_idx] = load_vector
            expected_P[6 * node_idx:6 * node_idx + 6] = load_vector
    P = fcn(nodal_loads, n_nodes)
    assert np.allclose(P, expected_P), 'Randomized loads should assemble correctly'
    nodal_loads = {0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]}
    P = fcn(nodal_loads, 2)
    assert P[0] == 1.0 and P[1] == 2.0 and (P[2] == 3.0), 'Node 0 DOF ordering [UX, UY, UZ, ...] should be correct'
    assert P[3] == 4.0 and P[4] == 5.0 and (P[5] == 6.0), 'Node 0 DOF ordering [..., RX, RY, RZ] should be correct'
    assert P[6] == 10.0 and P[7] == 20.0 and (P[8] == 30.0), 'Node 1 DOF ordering [UX, UY, UZ, ...] should be correct'
    assert P[9] == 40.0 and P[10] == 50.0 and (P[11] == 60.0), 'Node 1 DOF ordering [..., RX, RY, RZ] should be correct'