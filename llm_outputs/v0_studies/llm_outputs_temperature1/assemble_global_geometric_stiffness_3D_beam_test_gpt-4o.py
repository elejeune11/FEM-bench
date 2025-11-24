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
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}]
    u_global_zero = np.zeros(18)
    K_zero = fcn(node_coords, elements, u_global_zero)
    assert np.allclose(K_zero, np.zeros_like(K_zero))
    assert np.allclose(K_zero, K_zero.T)
    u_global_scaled = 2 * u_global_zero
    K_scaled = fcn(node_coords, elements, u_global_scaled)
    assert np.allclose(K_scaled, 2 * K_zero)
    u_global_1 = np.zeros(18)
    u_global_1[0] = 1.0
    u_global_2 = np.zeros(18)
    u_global_2[6] = 1.0
    K_1 = fcn(node_coords, elements, u_global_1)
    K_2 = fcn(node_coords, elements, u_global_2)
    K_superposed = fcn(node_coords, elements, u_global_1 + u_global_2)
    assert np.allclose(K_superposed, K_1 + K_2)
    elements_reversed = [{'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}]
    K_reversed = fcn(node_coords, elements_reversed, u_global_zero)
    assert np.allclose(K_zero, K_reversed)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}]
    u_global = np.random.rand(18)
    K_original = fcn(node_coords, elements, u_global)
    theta = np.pi / 2
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    node_coords_rotated = node_coords @ R.T
    u_global_rotated = np.zeros_like(u_global)
    for i in range(3):
        u_global_rotated[6 * i:6 * i + 3] = R @ u_global[6 * i:6 * i + 3]
        u_global_rotated[6 * i + 3:6 * i + 6] = R @ u_global[6 * i + 3:6 * i + 6]
    K_rotated = fcn(node_coords_rotated, elements, u_global_rotated)
    T = np.zeros((18, 18))
    for i in range(3):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    K_transformed = T @ K_original @ T.T
    assert np.allclose(K_rotated, K_transformed)