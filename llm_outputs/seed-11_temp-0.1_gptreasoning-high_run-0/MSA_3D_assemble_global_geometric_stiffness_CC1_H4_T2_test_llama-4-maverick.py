def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
    1) zero displacement produces a zero matrix,
    2) the assembled matrix is symmetric,
    3) scaling displacements scales K_g linearly,
    4) superposition holds for independent displacement states, and
    5) element order does not affect the assembled result."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    elements = [{'E': 1000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'E': 1000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}]
    u_global_zero = np.zeros(18)
    K_g_zero = fcn(node_coords, elements, u_global_zero)
    assert np.allclose(K_g_zero, np.zeros_like(K_g_zero))
    u_global = np.random.rand(18)
    K_g = fcn(node_coords, elements, u_global)
    assert np.allclose(K_g, K_g.T)
    scale = 2.0
    K_g_scaled = fcn(node_coords, elements, scale * u_global)
    assert np.allclose(K_g_scaled, scale * K_g)
    u_global2 = np.random.rand(18)
    K_g2 = fcn(node_coords, elements, u_global2)
    K_g_sum = fcn(node_coords, elements, u_global + u_global2)
    assert np.allclose(K_g_sum, K_g + K_g2)
    elements_reversed = list(reversed(elements))
    K_g_reversed = fcn(node_coords, elements_reversed, u_global)
    assert np.allclose(K_g_reversed, K_g)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    elements = [{'E': 1000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05, 'local_z': [0, 0, 1]}, {'E': 1000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05, 'local_z': [0, 0, 1]}]
    u_global = np.random.rand(18)
    R_global = R.from_euler('xyz', [np.pi / 4, np.pi / 3, np.pi / 2]).as_matrix()
    node_coords_rotated = np.dot(node_coords, R_global)
    elements_rotated = [{'E': e['E'], 'nu': e['nu'], 'A': e['A'], 'I_y': e['I_y'], 'I_z': e['I_z'], 'J': e['J'], 'local_z': np.dot(R_global, e['local_z'])} for e in elements]
    u_global_rotated = np.zeros_like(u_global)
    for i in range(3):
        u_global_rotated[i * 6:i * 6 + 3] = np.dot(R_global, u_global[i * 6:i * 6 + 3])
        u_global_rotated[i * 6 + 3:i * 6 + 6] = np.dot(R_global, u_global[i * 6 + 3:i * 6 + 6])
    K_g = fcn(node_coords, elements, u_global)
    K_g_rotated = fcn(node_coords_rotated, elements_rotated, u_global_rotated)
    T = np.kron(np.eye(3), np.block([[R_global, np.zeros((3, 3))], [np.zeros((3, 3)), R_global]]))
    assert np.allclose(K_g_rotated, np.dot(T, np.dot(K_g, T.T)))