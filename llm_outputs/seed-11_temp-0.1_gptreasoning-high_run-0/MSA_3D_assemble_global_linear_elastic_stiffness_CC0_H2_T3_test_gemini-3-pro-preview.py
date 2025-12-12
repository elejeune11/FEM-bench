def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    common_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0002, 'I_z': 0.0001, 'J': 5e-05, 'local_z': None}
    nodes_1 = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_1 = [common_props.copy()]
    elements_1[0].update({'node_i': 0, 'node_j': 1})
    K1 = fcn(nodes_1, elements_1)
    assert K1.shape == (12, 12)
    assert np.allclose(K1, K1.T, atol=1e-08)
    assert np.sum(np.abs(K1)) > 0
    assert np.all(np.diag(K1) >= 0)
    nodes_2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_2 = [common_props.copy(), common_props.copy()]
    elements_2[0].update({'node_i': 0, 'node_j': 1})
    elements_2[1].update({'node_i': 1, 'node_j': 2})
    K2 = fcn(nodes_2, elements_2)
    assert K2.shape == (18, 18)
    assert np.allclose(K2, K2.T, atol=1e-08)
    assert np.allclose(K2[0:6, 12:18], 0.0)
    assert np.sum(np.abs(K2[0:6, 6:12])) > 0
    assert np.sum(np.abs(K2[6:12, 12:18])) > 0
    nodes_3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elements_3 = [common_props.copy() for _ in range(3)]
    elements_3[0].update({'node_i': 0, 'node_j': 1})
    elements_3[1].update({'node_i': 1, 'node_j': 2})
    elements_3[2].update({'node_i': 2, 'node_j': 0})
    K3 = fcn(nodes_3, elements_3)
    assert K3.shape == (18, 18)
    assert np.allclose(K3, K3.T, atol=1e-08)
    assert np.sum(np.abs(K3[0:6, 6:12])) > 0
    assert np.sum(np.abs(K3[6:12, 12:18])) > 0
    assert np.sum(np.abs(K3[0:6, 12:18])) > 0
    nodes_4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_4 = [common_props.copy() for _ in range(4)]
    elements_4[0].update({'node_i': 0, 'node_j': 1})
    elements_4[1].update({'node_i': 1, 'node_j': 2})
    elements_4[2].update({'node_i': 2, 'node_j': 3})
    elements_4[3].update({'node_i': 3, 'node_j': 0})
    K4 = fcn(nodes_4, elements_4)
    assert K4.shape == (24, 24)
    assert np.allclose(K4, K4.T, atol=1e-08)
    assert np.allclose(K4[0:6, 12:18], 0.0)
    assert np.allclose(K4[6:12, 18:24], 0.0)
    assert np.sum(np.abs(K4[18:24, 0:6])) > 0