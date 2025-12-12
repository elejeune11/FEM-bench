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
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.333e-05
    I_z = 8.333e-05
    J = 0.0001667
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}]
    n_nodes = 3
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert K_g_zero.shape == (n_dof, n_dof)
    assert np.allclose(K_g_zero, 0.0, atol=1e-12)
    u_test = np.random.RandomState(42).randn(n_dof) * 0.01
    K_g = fcn(node_coords, elements, u_test)
    assert K_g.shape == (n_dof, n_dof)
    assert np.allclose(K_g, K_g.T, atol=1e-10)
    scale_factor = 2.5
    u_scaled = u_test * scale_factor
    K_g_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_g_scaled, K_g * scale_factor, atol=1e-10)
    u_1 = np.random.RandomState(43).randn(n_dof) * 0.01
    u_2 = np.random.RandomState(44).randn(n_dof) * 0.01
    K_g_1 = fcn(node_coords, elements, u_1)
    K_g_2 = fcn(node_coords, elements, u_2)
    K_g_sum = fcn(node_coords, elements, u_1 + u_2)
    assert np.allclose(K_g_sum, K_g_1 + K_g_2, atol=1e-10)
    elements_reversed = [elements[1], elements[0]]
    K_g_reversed = fcn(node_coords, elements_reversed, u_test)
    assert np.allclose(K_g, K_g_reversed, atol=1e-10)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.333e-05
    I_z = 8.333e-05
    J = 0.0001667
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}]
    n_nodes = 2
    n_dof = 6 * n_nodes
    u_original = np.random.RandomState(45).randn(n_dof) * 0.01
    K_g_original = fcn(node_coords, elements, u_original)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0.0], [np.sin(theta), np.cos(theta), 0.0], [0.0, 0.0, 1.0]])
    node_coords_rotated = node_coords @ R.T
    local_z_rotated = R @ np.array([0.0, 0.0, 1.0])
    elements_rotated = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z_rotated}]
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    u_rotated = T @ u_original
    K_g_rotated = fcn(node_coords_rotated, elements_rotated, u_rotated)
    K_g_transformed = T @ K_g_original @ T.T
    assert np.allclose(K_g_rotated, K_g_transformed, atol=1e-09)