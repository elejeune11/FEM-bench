def test_multi_element_core_correctness_assembly(fcn):
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1, 'I_rho': 1}, {'node_i': 1, 'node_j': 2, 'A': 1, 'I_rho': 1}]
    u_global = np.zeros(18)
    K = fcn(node_coords, elements, u_global)
    assert np.allclose(K, np.zeros((18, 18)))
    u_global = np.random.rand(18)
    K1 = fcn(node_coords, elements, u_global)
    K2 = fcn(node_coords, elements[::-1], u_global)
    assert np.allclose(K1, K2)
    assert np.allclose(K1, K1.T)
    u_global_scaled = 2 * u_global
    K_scaled = fcn(node_coords, elements, u_global_scaled)
    assert np.allclose(K_scaled, 2 * K1)
    u_global_2 = np.random.rand(18)
    K3 = fcn(node_coords, elements, u_global + u_global_2)
    assert np.allclose(K3, K1 + fcn(node_coords, elements, u_global_2))

def test_frame_objectivity_under_global_rotation(fcn):
    pass