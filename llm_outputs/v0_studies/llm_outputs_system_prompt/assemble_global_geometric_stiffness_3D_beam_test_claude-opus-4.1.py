def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06}]
    u_zero = np.zeros(18)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0), 'Zero displacement should produce zero geometric stiffness'
    np.random.seed(42)
    u_random = np.random.randn(18) * 0.01
    K_random = fcn(node_coords, elements, u_random)
    assert np.allclose(K_random, K_random.T), 'Geometric stiffness matrix must be symmetric'
    scale_factor = 2.5
    u_scaled = u_random * scale_factor
    K_scaled = fcn(node_coords, elements, u_scaled)
    K_expected_scaled = K_random * scale_factor
    assert np.allclose(K_scaled, K_expected_scaled, rtol=1e-10), 'K_g should scale linearly with displacements'
    u1 = np.zeros(18)
    u1[0] = 0.01
    u1[6] = -0.005
    K1 = fcn(node_coords, elements, u1)
    u2 = np.zeros(18)
    u2[2] = 0.008
    u2[14] = 0.003
    K2 = fcn(node_coords, elements, u2)
    u_combined = u1 + u2
    K_combined = fcn(node_coords, elements, u_combined)
    K_superposed = K1 + K2
    assert np.allclose(K_combined, K_superposed, rtol=1e-10), 'Superposition should hold for independent states'
    elements_reversed = [{'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06}, {'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06}]
    K_reversed = fcn(node_coords, elements_reversed, u_random)
    assert np.allclose(K_random, K_reversed), 'Element order should not affect assembled matrix'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.5, 0.0], [1.5, 1.0, 0.5]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06}, {'node_i': 2, 'node_j': 3, 'A': 0.01, 'I_rho': 1e-06}]
    np.random.seed(123)
    u_global = np.random.randn(24) * 0.01
    K_original = fcn(node_coords, elements, u_global)
    theta_z = np.pi / 4
    theta_y = np.pi / 6
    Rz = np.array([[np.cos(theta_z), -np.sin(theta_z), 0], [np.sin(theta_z), np.cos(theta_z), 0], [0, 0, 1]])
    Ry = np.array([[np.cos(theta_y), 0, np.sin(theta_y)], [0, 1, 0], [-np.sin(theta_y), 0, np.cos(theta_y)]])
    R = Ry @ Rz
    node_coords_rot = (R @ node_coords.T).T
    n_nodes = len(node_coords)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    u_global_rot = T @ u_global
    K_rotated = fcn(node_coords_rot, elements, u_global_rot)
    K_expected = T @ K_original @ T.T
    assert np.allclose(K_rotated, K_expected, rtol=1e-10, atol=1e-14), 'Frame objectivity violated: K_g^rot should equal T @ K_g @ T^T'