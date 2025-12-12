def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    """
    Comprehensive correctness test for MSA_3D_assemble_global_load_CC0_H0_T0:
    """
    nodal_loads = {0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    n_nodes = 1
    expected_output = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert np.allclose(fcn(nodal_loads, n_nodes), expected_output)
    nodal_loads = {0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0], 1: [7.0, 8.0, 9.0, 10.0, 11.0, 12.0]}
    n_nodes = 2
    expected_output = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    assert np.allclose(fcn(nodal_loads, n_nodes), expected_output)
    nodal_loads = {0: [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}
    n_nodes = 1
    expected_output = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(fcn(nodal_loads, n_nodes), expected_output)
    nodal_loads = {1: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    n_nodes = 3
    expected_output = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    assert np.allclose(fcn(nodal_loads, n_nodes), expected_output)
    np.random.seed(0)
    n_nodes = 10
    nodal_loads = {}
    for i in range(n_nodes):
        nodal_loads[i] = np.random.rand(6).tolist()
    output = fcn(nodal_loads, n_nodes)
    manual_output = np.zeros(6 * n_nodes)
    for (i, loads) in nodal_loads.items():
        manual_output[6 * i:6 * (i + 1)] = loads
    assert np.allclose(output, manual_output)