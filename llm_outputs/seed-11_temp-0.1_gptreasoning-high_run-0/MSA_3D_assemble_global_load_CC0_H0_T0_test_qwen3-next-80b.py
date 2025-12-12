def test_MSA_3D_assemble_global_load_CC0_H0_T0_comprehensive(fcn):
    import numpy as np
    from numpy.testing import assert_allclose
    nodal_loads = {0: [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]}
    n_nodes = 1
    P = fcn(nodal_loads, n_nodes)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert_allclose(P, expected, rtol=1e-10)
    nodal_loads = {0: [1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2: [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]}
    n_nodes = 3
    P = fcn(nodal_loads, n_nodes)
    expected = np.zeros(18)
    expected[0:6] = [1.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    expected[12:18] = [0.0, 2.0, 0.0, 0.0, 0.0, 0.0]
    assert_allclose(P, expected, rtol=1e-10)
    nodal_loads = {}
    n_nodes = 2
    P = fcn(nodal_loads, n_nodes)
    expected = np.zeros(12)
    assert_allclose(P, expected, rtol=1e-10)
    nodal_loads = {1: [10.0, 20.0, 30.0, 40.0, 50.0, 60.0], 5: [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]}
    n_nodes = 6
    P = fcn(nodal_loads, n_nodes)
    expected = np.zeros(36)
    expected[6:12] = [10.0, 20.0, 30.0, 40.0, 50.0, 60.0]
    expected[30:36] = [100.0, 200.0, 300.0, 400.0, 500.0, 600.0]
    assert_allclose(P, expected, rtol=1e-10)
    np.random.seed(42)
    n_nodes = 5
    nodal_loads = {}
    for i in range(n_nodes):
        if np.random.rand() > 0.5:
            nodal_loads[i] = np.random.rand(6).tolist()
    P = fcn(nodal_loads, n_nodes)
    manual = np.zeros(6 * n_nodes)
    for (node_idx, load_vec) in nodal_loads.items():
        manual[6 * node_idx:6 * node_idx + 6] = np.array(load_vec, dtype=float)
    assert_allclose(P, manual, rtol=1e-10)
    nodal_loads = {0: [1, 2, 3, 4, 5, 6]}
    n_nodes = 1
    P = fcn(nodal_loads, n_nodes)
    expected = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    assert_allclose(P, expected, rtol=1e-10)