def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    element_template = {'A': 0.01, 'I_rho': 0.0001, 'local_z': [0.0, 0.0, 1.0]}
    elements = [dict(node_i=0, node_j=1, **element_template), dict(node_i=1, node_j=2, **element_template)]
    elements_reversed = [dict(node_i=1, node_j=2, **element_template), dict(node_i=0, node_j=1, **element_template)]
    n_dof = len(nodes) * 6
    u_zero = np.zeros(n_dof)
    u1 = np.zeros(n_dof)
    u1[0] = 0.01
    u2 = np.zeros(n_dof)
    u2[7] = 0.02
    K_zero = fcn(nodes, elements, u_zero)
    assert_allclose(K_zero, np.zeros((n_dof, n_dof)), atol=1e-12)
    K1 = fcn(nodes, elements, u1)
    assert_allclose(K1, K1.T, rtol=1e-12)
    scale = 2.5
    K1_scaled = fcn(nodes, elements, scale * u1)
    assert_allclose(K1_scaled, scale * K1, rtol=1e-12)
    K2 = fcn(nodes, elements, u2)
    K_sum = fcn(nodes, elements, u1 + u2)
    assert_allclose(K_sum, K1 + K2, rtol=1e-12)
    K_normal = fcn(nodes, elements, u1)
    K_reversed = fcn(nodes, elements_reversed, u1)
    assert_allclose(K_normal, K_reversed, rtol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam."""
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [2.0, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 0.0001, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 0.0001, 'local_z': [0.0, 1.0, 0.0]}]
    n_nodes = len(nodes)
    n_dof = 6 * n_nodes
    u_global = np.random.rand(n_dof) * 0.01
    R = Rotation.random().as_matrix()
    nodes_rot = nodes @ R.T
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    u_rot = T @ u_global
    K_orig = fcn(nodes, elements, u_global)
    K_rot = fcn(nodes_rot, elements, u_rot)
    K_transformed = T @ K_orig @ T.T
    assert_allclose(K_rot, K_transformed, rtol=1e-10, atol=1e-10)