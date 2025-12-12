def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    nodal_loads_single = {0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    n_nodes_single = 1
    expected_single = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result_single = fcn(nodal_loads_single, n_nodes_single)
    assert np.allclose(result_single, expected_single), 'Single node load assembly failed'
    nodal_loads_multi = {0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2: [0.0, 0.0, 2.0, 0.0, 0.0, 0.0]}
    n_nodes_multi = 4
    expected_multi = np.zeros(6 * n_nodes_multi)
    expected_multi[0] = 1.0
    expected_multi[14] = 2.0
    result_multi = fcn(nodal_loads_multi, n_nodes_multi)
    assert np.allclose(result_multi, expected_multi), 'Multiple node load assembly with gaps failed'
    nodal_loads_zero = {}
    n_nodes_zero = 3
    expected_zero = np.zeros(6 * n_nodes_zero)
    result_zero = fcn(nodal_loads_zero, n_nodes_zero)
    assert np.allclose(result_zero, expected_zero), 'Zero load assembly failed'
    np.random.seed(42)
    n_nodes_rand = 5
    nodal_loads_rand = {i: np.random.rand(6).tolist() for i in range(n_nodes_rand) if np.random.rand() > 0.3}
    result_rand = fcn(nodal_loads_rand, n_nodes_rand)
    expected_rand = np.zeros(6 * n_nodes_rand)
    for (node_idx, load_vector) in nodal_loads_rand.items():
        start_idx = 6 * node_idx
        expected_rand[start_idx:start_idx + 6] = load_vector
    assert np.allclose(result_rand, expected_rand), 'Randomized load assembly failed'