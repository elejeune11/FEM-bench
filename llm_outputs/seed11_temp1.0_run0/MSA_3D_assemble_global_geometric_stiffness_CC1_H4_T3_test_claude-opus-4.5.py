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
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 0.0, 1.0])}
    elements = [{**elem_props, 'node_i': 0, 'node_j': 1}, {**elem_props, 'node_i': 1, 'node_j': 2}]
    n_nodes = 3
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert K_g_zero.shape == (n_dof, n_dof), 'Output shape mismatch'
    assert np.allclose(K_g_zero, 0.0, atol=1e-12), 'Zero displacement should produce zero K_g'
    np.random.seed(42)
    u_random = np.random.randn(n_dof) * 0.001
    K_g_random = fcn(node_coords, elements, u_random)
    assert np.allclose(K_g_random, K_g_random.T, atol=1e-10), 'K_g should be symmetric'
    scale_factor = 2.5
    u_scaled = scale_factor * u_random
    K_g_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_g_scaled, scale_factor * K_g_random, atol=1e-10), 'K_g should scale linearly with displacement'
    u_state1 = np.zeros(n_dof)
    u_state1[:6] = np.random.randn(6) * 0.001
    u_state2 = np.zeros(n_dof)
    u_state2[12:18] = np.random.randn(6) * 0.001
    K_g_state1 = fcn(node_coords, elements, u_state1)
    K_g_state2 = fcn(node_coords, elements, u_state2)
    K_g_combined = fcn(node_coords, elements, u_state1 + u_state2)
    assert np.allclose(K_g_combined, K_g_state1 + K_g_state2, atol=1e-10), 'Superposition should hold for K_g'
    elements_reversed = [{**elem_props, 'node_i': 1, 'node_j': 2}, {**elem_props, 'node_i': 0, 'node_j': 1}]
    K_g_reversed = fcn(node_coords, elements_reversed, u_random)
    assert np.allclose(K_g_random, K_g_reversed, atol=1e-10), 'Element order should not affect assembled K_g'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords_orig = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 0.0, 1.0])}
    elements_orig = [{**elem_props, 'node_i': 0, 'node_j': 1}]
    n_nodes = 2
    n_dof = 6 * n_nodes
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    node_coords_rot = (R @ node_coords_orig.T).T
    local_z_rot = R @ elem_props['local_z']
    elements_rot = [{**elem_props, 'node_i': 0, 'node_j': 1, 'local_z': local_z_rot}]
    np.random.seed(123)
    u_orig = np.random.randn(n_dof) * 0.001

    def build_T_matrix(R, n_nodes):
        """Build block-diagonal transformation matrix."""
        n_dof = 6 * n_nodes
        T = np.zeros((n_dof, n_dof))
        for i in range(n_nodes):
            idx = 6 * i
            T[idx:idx + 3, idx:idx + 3] = R
            T[idx + 3:idx + 6, idx + 3:idx + 6] = R
        return T
    T = build_T_matrix(R, n_nodes)
    u_rot = T @ u_orig
    K_g_orig = fcn(node_coords_orig, elements_orig, u_orig)
    K_g_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_g_expected = T @ K_g_orig @ T.T
    assert np.allclose(K_g_rot, K_g_expected, atol=1e-08), 'K_g should transform as K_g^rot = T @ K_g @ T^T under global rotation'
    assert np.allclose(K_g_orig, K_g_orig.T, atol=1e-10), 'Original K_g should be symmetric'
    assert np.allclose(K_g_rot, K_g_rot.T, atol=1e-10), 'Rotated K_g should be symmetric'
    phi = np.pi / 2
    R2 = np.array([[np.cos(phi), 0.0, np.sin(phi)], [0.0, 1.0, 0.0], [-np.sin(phi), 0.0, np.cos(phi)]])
    node_coords_rot2 = (R2 @ node_coords_orig.T).T
    local_z_rot2 = R2 @ elem_props['local_z']
    elements_rot2 = [{**elem_props, 'node_i': 0, 'node_j': 1, 'local_z': local_z_rot2}]
    T2 = build_T_matrix(R2, n_nodes)
    u_rot2 = T2 @ u_orig
    K_g_rot2 = fcn(node_coords_rot2, elements_rot2, u_rot2)
    K_g_expected2 = T2 @ K_g_orig @ T2.T
    assert np.allclose(K_g_rot2, K_g_expected2, atol=1e-08), 'Frame objectivity should hold for arbitrary rotations'