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
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-05}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-05}]
    n_nodes = 3
    n_dofs = 6 * n_nodes
    u_zero = np.zeros(n_dofs)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert K_g_zero.shape == (n_dofs, n_dofs), 'K_g shape mismatch'
    assert np.allclose(K_g_zero, 0.0, atol=1e-12), 'Zero displacement should produce zero K_g'
    u_test = np.random.randn(n_dofs) * 0.01
    K_g = fcn(node_coords, elements, u_test)
    assert np.allclose(K_g, K_g.T, atol=1e-10), 'K_g must be symmetric'
    scale_factors = [0.5, 2.0, 3.0]
    K_g_base = fcn(node_coords, elements, u_test)
    for scale in scale_factors:
        K_g_scaled = fcn(node_coords, elements, scale * u_test)
        expected_K_g = scale * K_g_base
        assert np.allclose(K_g_scaled, expected_K_g, atol=1e-10), f'K_g should scale linearly with displacement scaling factor {scale}'
    u1 = np.zeros(n_dofs)
    u1[0] = 0.01
    u2 = np.zeros(n_dofs)
    u2[1] = 0.01
    K_g_1 = fcn(node_coords, elements, u1)
    K_g_2 = fcn(node_coords, elements, u2)
    K_g_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_g_sum, K_g_1 + K_g_2, atol=1e-10), 'Superposition should hold for independent displacement states'
    elements_reversed = [{'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-05}, {'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-05}]
    u_test2 = np.random.randn(n_dofs) * 0.01
    K_g_ordered = fcn(node_coords, elements, u_test2)
    K_g_reversed = fcn(node_coords, elements_reversed, u_test2)
    assert np.allclose(K_g_ordered, K_g_reversed, atol=1e-10), 'Element order should not affect assembled K_g'

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-05}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-05}]
    n_nodes = 3
    n_dofs = 6 * n_nodes
    u_original = np.random.randn(n_dofs) * 0.005
    K_g_original = fcn(node_coords, elements, u_original)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]], dtype=float)
    node_coords_rotated = node_coords @ R.T
    u_rotated = np.zeros_like(u_original)
    for i in range(n_nodes):
        u_trans = u_original[6 * i:6 * i + 3]
        u_rotated[6 * i:6 * i + 3] = R @ u_trans
        u_rot = u_original[6 * i + 3:6 * i + 6]
        u_rotated[6 * i + 3:6 * i + 6] = R @ u_rot
    K_g_rotated = fcn(node_coords_rotated, elements, u_rotated)
    T = np.zeros((n_dofs, n_dofs), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    K_g_expected = T @ K_g_original @ T.T
    assert np.allclose(K_g_rotated, K_g_expected, atol=1e-08), 'Geometric stiffness matrix should satisfy frame objectivity under global rotation'