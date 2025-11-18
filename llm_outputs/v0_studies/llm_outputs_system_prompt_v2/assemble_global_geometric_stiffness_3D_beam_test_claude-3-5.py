def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 0.0001}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 0.0001}]
    n_dof = 6 * len(nodes)
    u_zero = np.zeros(n_dof)
    K_zero = fcn(nodes, elements, u_zero)
    assert_allclose(K_zero, np.zeros((n_dof, n_dof)))
    u_test = np.random.rand(n_dof)
    K = fcn(nodes, elements, u_test)
    assert_allclose(K, K.T, rtol=1e-12)
    scale = 2.5
    K_scaled = fcn(nodes, elements, scale * u_test)
    assert_allclose(K_scaled, scale * K, rtol=1e-12)
    u1 = np.random.rand(n_dof)
    u2 = np.random.rand(n_dof)
    K1 = fcn(nodes, elements, u1)
    K2 = fcn(nodes, elements, u2)
    K_sum = fcn(nodes, elements, u1 + u2)
    assert_allclose(K_sum, K1 + K2, rtol=1e-12)
    elements_reversed = elements[::-1]
    K_rev = fcn(nodes, elements_reversed, u_test)
    assert_allclose(K, K_rev, rtol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 0.0001}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 0.0001}]
    n_nodes = len(nodes)
    n_dof = 6 * n_nodes
    u_test = np.random.rand(n_dof)
    K_orig = fcn(nodes, elements, u_test)
    R = Rotation.random().as_matrix()
    nodes_rot = nodes @ R.T
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    u_rot = T @ u_test
    K_rot = fcn(nodes_rot, elements, u_rot)
    K_transformed = T @ K_orig @ T.T
    assert_allclose(K_rot, K_transformed, rtol=1e-10)