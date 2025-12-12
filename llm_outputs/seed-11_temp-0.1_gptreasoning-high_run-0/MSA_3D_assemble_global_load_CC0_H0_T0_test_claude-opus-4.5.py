def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    n_nodes = 5
    nodal_loads = {}
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (6 * n_nodes,), 'Output shape should be (6 * n_nodes,)'
    assert np.allclose(P, np.zeros(6 * n_nodes)), 'Empty loads should produce zero vector'
    n_nodes = 3
    load_vec = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    nodal_loads = {0: load_vec}
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (18,), 'Output shape should be 18 for 3 nodes'
    assert np.allclose(P[0:6], load_vec), 'DOFs 0-5 should match load at node 0'
    assert np.allclose(P[6:], np.zeros(12)), 'Remaining DOFs should be zero'
    n_nodes = 4
    load_vec = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    nodal_loads = {3: load_vec}
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (24,), 'Output shape should be 24 for 4 nodes'
    assert np.allclose(P[0:18], np.zeros(18)), 'DOFs 0-17 should be zero'
    assert np.allclose(P[18:24], load_vec), 'DOFs 18-23 should match load at node 3'
    n_nodes = 5
    load_node_1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    load_node_3 = [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    nodal_loads = {1: load_node_1, 3: load_node_3}
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (30,), 'Output shape should be 30 for 5 nodes'
    assert np.allclose(P[0:6], np.zeros(6)), 'Node 0 should have zero loads'
    assert np.allclose(P[6:12], load_node_1), 'Node 1 should have specified loads'
    assert np.allclose(P[12:18], np.zeros(6)), 'Node 2 should have zero loads'
    assert np.allclose(P[18:24], load_node_3), 'Node 3 should have specified loads'
    assert np.allclose(P[24:30], np.zeros(6)), 'Node 4 should have zero loads'
    n_nodes = 2
    load_node_0 = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    load_node_1 = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    nodal_loads = {0: load_node_0, 1: load_node_1}
    P = fcn(nodal_loads, n_nodes)
    for i in range(6):
        assert P[i] == load_node_0[i], f'DOF {i} should match load_node_0[{i}]'
        assert P[6 + i] == load_node_1[i], f'DOF {6 + i} should match load_node_1[{i}]'
    n_nodes = 2
    load_vec_int = [1, 2, 3, 4, 5, 6]
    nodal_loads = {0: load_vec_int}
    P = fcn(nodal_loads, n_nodes)
    assert P.dtype == np.float64 or np.issubdtype(P.dtype, np.floating), 'Output should be float'
    assert np.allclose(P[0:6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 'Integer inputs should convert to float'
    np.random.seed(42)
    n_nodes = 10
    num_loaded_nodes = 5
    loaded_node_indices = np.random.choice(n_nodes, num_loaded_nodes, replace=False)
    nodal_loads = {}
    for idx in loaded_node_indices:
        nodal_loads[int(idx)] = np.random.randn(6).tolist()
    P = fcn(nodal_loads, n_nodes)
    P_ref = np.zeros(6 * n_nodes)
    for (node_idx, load_vec) in nodal_loads.items():
        start_dof = 6 * node_idx
        P_ref[start_dof:start_dof + 6] = load_vec
    assert P.shape == P_ref.shape, 'Shape should match reference'
    assert np.allclose(P, P_ref), 'Assembled vector should match manual reference'
    n_nodes = 1
    load_vec = [5.5, -3.2, 7.8, 1.1, -2.2, 3.3]
    nodal_loads = {0: load_vec}
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (6,), 'Single node should have 6 DOFs'
    assert np.allclose(P, load_vec), 'Single node load should match input'
    n_nodes = 3
    load_vec_np = np.array([1.5, 2.5, 3.5, 4.5, 5.5, 6.5])
    nodal_loads = {1: load_vec_np}
    P = fcn(nodal_loads, n_nodes)
    assert np.allclose(P[6:12], load_vec_np), 'Numpy array input should work correctly'
    n_nodes = 2
    load_vec = [-1.0, 2.0, -3.0, 4.0, -5.0, 6.0]
    nodal_loads = {0: load_vec}
    P = fcn(nodal_loads, n_nodes)
    assert np.allclose(P[0:6], load_vec), 'Negative and mixed sign values should be preserved'