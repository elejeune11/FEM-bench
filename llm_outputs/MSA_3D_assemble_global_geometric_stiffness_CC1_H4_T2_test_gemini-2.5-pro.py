def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003, 'local_z': [0.0, 0.0, 1.0]}
    elements = [{'nodes': [0, 1], **elem_props}, {'nodes': [1, 2], **elem_props}]
    elements_rev = [{'nodes': [1, 2], **elem_props}, {'nodes': [0, 1], **elem_props}]
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_g_zero, np.zeros((n_dof, n_dof)))
    rng = np.random.default_rng(seed=42)
    u_rand1 = rng.random(n_dof)
    u_rand2 = rng.random(n_dof)
    K_g1 = fcn(node_coords, elements, u_rand1)
    assert K_g1.shape == (n_dof, n_dof)
    assert np.allclose(K_g1, K_g1.T)
    alpha = 3.14
    K_g_scaled = fcn(node_coords, elements, alpha * u_rand1)
    assert np.allclose(K_g_scaled, alpha * K_g1)
    K_g2 = fcn(node_coords, elements, u_rand2)
    K_g_sum_u = fcn(node_coords, elements, u_rand1 + u_rand2)
    assert np.allclose(K_g_sum_u, K_g1 + K_g2)
    K_g_rev = fcn(node_coords, elements_rev, u_rand1)
    assert np.allclose(K_g1, K_g_rev)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [5.0, 0.0, 0.0]])
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    local_z_vec = np.array([0.0, 1.0, 0.0])
    elements = [{'nodes': [0, 1], 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003, 'local_z': local_z_vec}]
    rng = np.random.default_rng(seed=123)
    u_global = rng.random(n_dof)
    K_g_orig = fcn(node_coords, elements, u_global)
    R = Rotation.from_euler('zyx', [20, -30, 50], degrees=True).as_matrix()
    node_coords_rot = (R @ node_coords.T).T
    local_z_rot = R @ local_z_vec
    elements_rot = [{'nodes': [0, 1], 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003, 'local_z': local_z_rot}]
    u_reshaped = u_global.reshape((n_nodes, 6))
    u_rot_reshaped = np.zeros_like(u_reshaped)
    for i in range(n_nodes):
        u_rot_reshaped[i, 0:3] = R @ u_reshaped[i, 0:3]
        u_rot_reshaped[i, 3:6] = R @ u_reshaped[i, 3:6]
    u_global_rot = u_rot_reshaped.flatten()
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    T_node = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    T = block_diag(*[T_node] * n_nodes)
    K_g_transformed = T @ K_g_orig @ T.T
    assert np.allclose(K_g_rot, K_g_transformed, atol=1e-09)