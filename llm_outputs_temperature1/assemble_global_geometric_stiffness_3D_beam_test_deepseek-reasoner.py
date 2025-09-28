def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-05, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-05, 'local_z': [0, 0, 1]}]
    n_nodes = 3
    dof_per_node = 6
    total_dof = n_nodes * dof_per_node
    u_zero = np.zeros(total_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_g_zero, 0.0), 'Zero displacement should produce zero geometric stiffness'
    u_test = np.random.rand(total_dof) * 0.01
    K_g = fcn(node_coords, elements, u_test)
    assert np.allclose(K_g, K_g.T), 'Geometric stiffness matrix must be symmetric'
    scale_factor = 2.5
    K_g_scaled = fcn(node_coords, elements, scale_factor * u_test)
    assert np.allclose(K_g_scaled, scale_factor * K_g), 'K_g should scale linearly with displacement'
    u1 = np.random.rand(total_dof) * 0.01
    u2 = np.random.rand(total_dof) * 0.01
    K_g_u1 = fcn(node_coords, elements, u1)
    K_g_u2 = fcn(node_coords, elements, u2)
    K_g_u1_plus_u2 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_g_u1_plus_u2, K_g_u1 + K_g_u2), 'K_g should satisfy superposition'
    elements_reversed = list(reversed(elements))
    K_g_reversed = fcn(node_coords, elements_reversed, u_test)
    assert np.allclose(K_g_reversed, K_g), 'Assembly should be invariant to element order'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.2], [1.5, 1.0, 0.4]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-05, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-05, 'local_z': [0, 0, 1]}]
    n_nodes = 3
    dof_per_node = 6
    total_dof = n_nodes * dof_per_node
    u_global = np.random.rand(total_dof) * 0.01
    K_g_orig = fcn(node_coords, elements, u_global)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for ele in elements:
        ele_rot = ele.copy()
        if 'local_z' in ele:
            ele_rot['local_z'] = list(np.array(ele['local_z']) @ R.T)
        elements_rot.append(ele_rot)
    u_global_rot = np.zeros_like(u_global)
    for i in range(n_nodes):
        u_global_rot[i * dof_per_node:i * dof_per_node + 3] = R @ u_global[i * dof_per_node:i * dof_per_node + 3]
        u_global_rot[i * dof_per_node + 3:i * dof_per_node + 6] = R @ u_global[i * dof_per_node + 3:i * dof_per_node + 6]
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    T_blocks = []
    for _ in range(n_nodes):
        T_node = np.zeros((6, 6))
        T_node[:3, :3] = R
        T_node[3:, 3:] = R
        T_blocks.append(T_node)
    T = np.kron(np.eye(n_nodes), np.eye(6))
    for (i, block) in enumerate(T_blocks):
        start_idx = i * 6
        end_idx = (i + 1) * 6
        T[start_idx:end_idx, start_idx:end_idx] = block
    K_g_transformed = T @ K_g_orig @ T.T
    assert np.allclose(K_g_rot, K_g_transformed, atol=1e-10), 'Geometric stiffness should transform correctly under rigid body rotation'