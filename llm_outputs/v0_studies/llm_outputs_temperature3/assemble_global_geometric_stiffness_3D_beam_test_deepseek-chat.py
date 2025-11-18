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
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}]
    n_nodes = 3
    dof_per_node = 6
    total_dof = n_nodes * dof_per_node
    u_zero = np.zeros(total_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_g_zero, 0.0), 'Zero displacement should produce zero geometric stiffness'
    u_test = np.random.rand(total_dof) * 0.01
    K_g = fcn(node_coords, elements, u_test)
    assert np.allclose(K_g, K_g.T), 'Geometric stiffness matrix should be symmetric'
    scale = 2.0
    K_g_scaled = fcn(node_coords, elements, scale * u_test)
    assert np.allclose(K_g_scaled, scale * K_g), 'K_g should scale linearly with displacement'
    u1 = np.random.rand(total_dof) * 0.01
    u2 = np.random.rand(total_dof) * 0.01
    K_g1 = fcn(node_coords, elements, u1)
    K_g2 = fcn(node_coords, elements, u2)
    K_g_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_g_sum, K_g1 + K_g2), 'Superposition should hold for geometric stiffness'
    elements_reversed = [{'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}, {'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}]
    K_g_reversed = fcn(node_coords, elements_reversed, u_test)
    assert np.allclose(K_g, K_g_reversed), 'Element order should not affect assembled result'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}]
    n_nodes = 2
    dof_per_node = 6
    total_dof = n_nodes * dof_per_node
    u_global = np.random.rand(total_dof) * 0.01
    K_g_original = fcn(node_coords, elements, u_global)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    T = np.zeros((total_dof, total_dof))
    for i in range(n_nodes):
        start_idx = i * dof_per_node
        T[start_idx:start_idx + 3, start_idx:start_idx + 3] = R
        T[start_idx + 3:start_idx + 6, start_idx + 3:start_idx + 6] = R
    node_coords_rotated = node_coords @ R.T
    u_rotated = T @ u_global
    K_g_rotated = fcn(node_coords_rotated, elements, u_rotated)
    K_g_expected = T @ K_g_original @ T.T
    assert np.allclose(K_g_rotated, K_g_expected, atol=1e-10), 'Geometric stiffness should transform correctly under global rotation'