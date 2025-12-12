def test_multi_element_core_correctness_assembly(fcn: Callable) -> None:
    """
    Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
    1) zero displacement produces a zero matrix,
    2) the assembled matrix is symmetric,
    3) scaling displacements scales K_g linearly,
    4) superposition holds for independent displacement states, and
    5) element order does not affect the assembled result.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 0.0001, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 0.0001, 'local_z': None}]
    n_nodes = 3
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert K_g_zero.shape == (n_dof, n_dof), 'Output shape mismatch'
    assert np.allclose(K_g_zero, 0.0, atol=1e-12), 'Zero displacement should produce zero K_g'
    u_test = np.random.randn(n_dof) * 0.01
    K_g = fcn(node_coords, elements, u_test)
    assert np.allclose(K_g, K_g.T, atol=1e-10), 'K_g must be symmetric'
    scale_factor = 2.5
    u_scaled = u_test * scale_factor
    K_g_scaled = fcn(node_coords, elements, u_scaled)
    K_g_expected_scaled = K_g * scale_factor ** 2
    assert np.allclose(K_g_scaled, K_g_expected_scaled, rtol=1e-09), 'K_g should scale quadratically with displacement magnitude'
    u1 = np.zeros(n_dof)
    u1[0] = 0.01
    u2 = np.zeros(n_dof)
    u2[6] = 0.01
    u_combined = u1 + u2
    K_g_1 = fcn(node_coords, elements, u1)
    K_g_2 = fcn(node_coords, elements, u2)
    K_g_combined = fcn(node_coords, elements, u_combined)
    K_g_superposed = K_g_1 + K_g_2
    assert np.allclose(K_g_combined, K_g_superposed, rtol=1e-08), 'Superposition should hold for small displacements'
    elements_reversed = [elements[1], elements[0]]
    K_g_reversed = fcn(node_coords, elements_reversed, u_test)
    K_g_original = fcn(node_coords, elements, u_test)
    assert np.allclose(K_g_reversed, K_g_original, atol=1e-10), 'Element order should not affect assembled K_g'

def test_frame_objectivity_under_global_rotation(fcn: Callable) -> None:
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 0.0001, 'local_z': None}]
    n_nodes = 2
    n_dof = 6 * n_nodes
    u_original = np.array([0.001, 0.002, 0.0005, 0.0001, 0.0002, 0.0, -0.001, 0.0015, 0.0003, 0.00015, -0.0001, 0.0])
    K_g_original = fcn(node_coords, elements, u_original)
    angle_deg = 45.0
    rotation = Rotation.from_euler('z', angle_deg, degrees=True)
    R = rotation.as_matrix()
    node_coords_rotated = node_coords @ R.T
    T_block = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        T_block[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T_block[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    u_rotated = T_block @ u_original
    elements_rotated = [{'node_i': e['node_i'], 'node_j': e['node_j'], 'A': e['A'], 'I_rho': e['I_rho'], 'local_z': e.get('local_z')} for e in elements]
    K_g_rotated = fcn(node_coords_rotated, elements_rotated, u_rotated)
    K_g_expected = T_block @ K_g_original @ T_block.T
    assert np.allclose(K_g_rotated, K_g_expected, rtol=1e-07, atol=1e-10), 'Geometric stiffness matrix should satisfy frame objectivity: K_g^rot = T K_g T^T'