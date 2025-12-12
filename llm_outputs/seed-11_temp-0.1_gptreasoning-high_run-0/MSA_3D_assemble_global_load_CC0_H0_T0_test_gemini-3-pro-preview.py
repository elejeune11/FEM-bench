def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    n_nodes = 10
    nodal_loads = {}
    P_zeros = fcn(nodal_loads, n_nodes)
    assert isinstance(P_zeros, np.ndarray)
    assert P_zeros.shape == (60,)
    assert np.allclose(P_zeros, 0.0), 'Expected zero vector when nodal_loads is empty'
    n_nodes = 3
    load_0 = [10.0, -5.0, 0.0, 1.0, 0.0, -1.0]
    load_2 = [0.0, 20.0, 15.0, 0.0, 5.0, 5.0]
    nodal_loads = {0: load_0, 2: load_2}
    P_det = fcn(nodal_loads, n_nodes)
    assert P_det.shape == (18,)
    np.testing.assert_allclose(P_det[0:6], load_0, err_msg='Mismatch at Node 0')
    np.testing.assert_allclose(P_det[6:12], 0.0, err_msg='Mismatch at Node 1 (expected zeros)')
    np.testing.assert_allclose(P_det[12:18], load_2, err_msg='Mismatch at Node 2')
    rng = np.random.default_rng(42)
    n_nodes_rand = 20
    num_dofs = n_nodes_rand * 6
    nodal_loads_rand = {}
    expected_P = np.zeros(num_dofs)
    active_nodes = rng.choice(n_nodes_rand, size=10, replace=False)
    for node_idx in active_nodes:
        load_vec = rng.standard_normal(6)
        nodal_loads_rand[node_idx] = load_vec
        start_idx = node_idx * 6
        expected_P[start_idx:start_idx + 6] = load_vec
    result_P = fcn(nodal_loads_rand, n_nodes_rand)
    assert result_P.shape == (num_dofs,)
    np.testing.assert_allclose(result_P, expected_P, err_msg='Randomized assembly does not match manual construction')