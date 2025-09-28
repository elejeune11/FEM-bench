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
    n_dof = n_nodes * dof_per_node
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert K_g_zero.shape == (n_dof, n_dof)
    assert np.allclose(K_g_zero, 0.0)
    u_test = np.random.rand(n_dof) * 0.01
    K_g = fcn(node_coords, elements, u_test)
    assert np.allclose(K_g, K_g.T)
    scale = 2.5
    u_scaled = u_test * scale
    K_g_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_g_scaled, K_g * scale, rtol=1e-10)
    u1 = np.random.rand(n_dof) * 0.01
    u2 = np.random.rand(n_dof) * 0.01
    K_g_u1 = fcn(node_coords, elements, u1)
    K_g_u2 = fcn(node_coords, elements, u2)
    K_g_u1_plus_u2 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_g_u1_plus_u2, K_g_u1 + K_g_u2, rtol=1e-10)
    elements_reversed = list(reversed(elements))
    K_g_reversed = fcn(node_coords, elements_reversed, u_test)
    assert np.allclose(K_g, K_g_reversed)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-05, 'local_z': [0, 0, 1]}]
    n_nodes = 2
    dof_per_node = 6
    n_dof = n_nodes * dof_per_node
    u_global = np.random.rand(n_dof) * 0.01
    K_g_original = fcn(node_coords, elements, u_global)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    node_coords_rotated = node_coords @ R.T
    elements_rotated = []
    for ele in elements:
        ele_copy = ele.copy()
        if 'local_z' in ele:
            local_z_rotated = np.array(ele['local_z']) @ R.T
            ele_copy['local_z'] = local_z_rotated.tolist()
        elements_rotated.append(ele_copy)
    T_block = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    T_global = np.kron(np.eye(n_nodes), T_block)
    u_global_rotated = T_global @ u_global
    K_g_rotated = fcn(node_coords_rotated, elements_rotated, u_global_rotated)
    K_g_expected = T_global @ K_g_original @ T_global.T
    assert np.allclose(K_g_rotated, K_g_expected, rtol=1e-10)