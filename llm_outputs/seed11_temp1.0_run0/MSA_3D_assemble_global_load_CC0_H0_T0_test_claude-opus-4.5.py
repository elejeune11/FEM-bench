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
    assert P.shape == (6 * n_nodes,), 'Output shape should be (6 * n_nodes,)'
    assert np.allclose(P[0:6], load_vec), 'Load at node 0 should be at indices 0-5'
    assert np.allclose(P[6:], np.zeros(12)), 'Other nodes should have zero loads'
    n_nodes = 4
    load_vec = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    nodal_loads = {3: load_vec}
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (6 * n_nodes,), 'Output shape should be (6 * n_nodes,)'
    assert np.allclose(P[18:24], load_vec), 'Load at node 3 should be at indices 18-23'
    assert np.allclose(P[0:18], np.zeros(18)), 'Other nodes should have zero loads'
    n_nodes = 5
    load_node_0 = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    load_node_2 = [0.0, 100.0, 0.0, 0.0, 0.0, 0.0]
    load_node_4 = [0.0, 0.0, 0.0, 50.0, 60.0, 70.0]
    nodal_loads = {0: load_node_0, 2: load_node_2, 4: load_node_4}
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (6 * n_nodes,), 'Output shape should be (6 * n_nodes,)'
    assert np.allclose(P[0:6], load_node_0), 'Load at node 0 incorrect'
    assert np.allclose(P[6:12], np.zeros(6)), 'Node 1 should have zero loads'
    assert np.allclose(P[12:18], load_node_2), 'Load at node 2 incorrect'
    assert np.allclose(P[18:24], np.zeros(6)), 'Node 3 should have zero loads'
    assert np.allclose(P[24:30], load_node_4), 'Load at node 4 incorrect'
    n_nodes = 3
    load_node_1 = [11.0, 22.0, 33.0, 44.0, 55.0, 66.0]
    nodal_loads = {1: load_node_1}
    P = fcn(nodal_loads, n_nodes)
    for i in range(6):
        assert P[6 * 1 + i] == load_node_1[i], f'DOF {6 * 1 + i} should equal load_node_1[{i}]'
    n_nodes = 2
    load_vec_int = [1, 2, 3, 4, 5, 6]
    nodal_loads = {0: load_vec_int}
    P = fcn(nodal_loads, n_nodes)
    assert P.dtype == np.float64 or np.issubdtype(P.dtype, np.floating), 'Output should be float'
    assert np.allclose(P[0:6], [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]), 'Integer input should be converted to float'
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
        P_ref[start_dof:start_dof + 6] = np.array(load_vec, dtype=float)
    assert P.shape == P_ref.shape, 'Shape mismatch with reference'
    assert np.allclose(P, P_ref), 'Load vector does not match manual reference assembly'
    n_nodes = 2
    load_vec_np = np.array([7.5, 8.5, 9.5, 10.5, 11.5, 12.5])
    nodal_loads = {1: load_vec_np}
    P = fcn(nodal_loads, n_nodes)
    assert np.allclose(P[6:12], load_vec_np), 'Numpy array input should work correctly'
    assert np.allclose(P[0:6], np.zeros(6)), 'Node 0 should have zero loads'
    n_nodes = 1
    load_vec = [100.0, -200.0, 300.0, -400.0, 500.0, -600.0]
    nodal_loads = {0: load_vec}
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (6,), 'Single node should produce 6-element vector'
    assert np.allclose(P, load_vec), 'Single node load incorrect'
    n_nodes = 100
    nodal_loads = {}
    P = fcn(nodal_loads, n_nodes)
    assert P.shape == (600,), 'Large unloaded structure should have correct shape'
    assert np.all(P == 0.0), 'Unloaded structure should have all zero loads'