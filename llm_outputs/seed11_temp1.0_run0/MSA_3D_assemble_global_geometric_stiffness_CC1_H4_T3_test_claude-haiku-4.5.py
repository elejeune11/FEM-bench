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
    elements = [{'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0.0, 0.0, 1.0])}, {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0.0, 0.0, 1.0])}]
    n_nodes = 3
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert K_g_zero.shape == (n_dof, n_dof)
    assert np.allclose(K_g_zero, 0.0, atol=1e-12)
    u_test = np.random.RandomState(42).randn(n_dof) * 0.001
    K_g = fcn(node_coords, elements, u_test)
    assert K_g.shape == (n_dof, n_dof)
    assert np.allclose(K_g, K_g.T, rtol=1e-10)
    u_1 = np.random.RandomState(43).randn(n_dof) * 0.001
    K_g_1 = fcn(node_coords, elements, u_1)
    scale = 2.5
    K_g_scaled = fcn(node_coords, elements, scale * u_1)
    assert np.allclose(K_g_scaled, scale * K_g_1, rtol=1e-10)
    u_a = np.random.RandomState(44).randn(n_dof) * 0.001
    u_b = np.random.RandomState(45).randn(n_dof) * 0.001
    K_g_a = fcn(node_coords, elements, u_a)
    K_g_b = fcn(node_coords, elements, u_b)
    K_g_sum = fcn(node_coords, elements, u_a + u_b)
    assert np.allclose(K_g_sum, K_g_a + K_g_b, rtol=1e-10)
    elements_reversed = [elements[1], elements[0]]
    K_g_original = fcn(node_coords, elements, u_test)
    K_g_reversed = fcn(node_coords, elements_reversed, u_test)
    assert np.allclose(K_g_original, K_g_reversed, rtol=1e-10)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x, u_y, u_z, rx, ry, rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    elements = [{'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0.0, 0.0, 1.0])}]
    n_nodes = 2
    n_dof = 6 * n_nodes
    u_original = np.random.RandomState(50).randn(n_dof) * 0.001
    K_g_original = fcn(node_coords, elements, u_original)
    angle = np.pi / 4
    c = np.cos(angle)
    s = np.sin(angle)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    node_coords_rot = node_coords @ R.T
    u_rot = np.zeros_like(u_original)
    for i in range(n_nodes):
        u_rot[6 * i:6 * i + 3] = R @ u_original[6 * i:6 * i + 3]
        u_rot[6 * i + 3:6 * i + 6] = R @ u_original[6 * i + 3:6 * i + 6]
    elements_rot = []
    for elem in elements:
        elem_copy = elem.copy()
        elem_copy['local_z'] = R @ elem['local_z']
        elements_rot.append(elem_copy)
    K_g_rot = fcn(node_coords_rot, elements_rot, u_rot)
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    K_g_transformed = T @ K_g_original @ T.T
    assert np.allclose(K_g_rot, K_g_transformed, rtol=1e-09, atol=1e-14)