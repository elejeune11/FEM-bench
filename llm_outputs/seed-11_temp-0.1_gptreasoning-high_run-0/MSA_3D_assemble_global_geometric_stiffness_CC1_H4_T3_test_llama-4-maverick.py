def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
    1) zero displacement produces a zero matrix,
    2) the assembled matrix is symmetric,
    3) scaling displacements scales K_g linearly,
    4) superposition holds for independent displacement states, and
    5) element order does not affect the assembled result."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0001}, {'node_i': 1, 'node_j': 2, 'E': 1000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0001}]
    u_global_zero = np.zeros(18)
    K_g_zero = fcn(node_coords, elements, u_global_zero)
    assert np.allclose(K_g_zero, 0)
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
    elements_reversed = elements[::-1]
    K_g_reversed = fcn(node_coords, elements_reversed, u_global)
    assert np.allclose(K_g_reversed, K_g)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0001}, {'node_i': 1, 'node_j': 2, 'E': 1000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0001}]
    u_global = np.random.rand(18)
    rot = R.from_euler('xyz', np.random.rand(3), degrees=True)
    R_mat = rot.as_matrix()
    node_coords_rot = np.dot(node_coords, R_mat.T)
    elements_rot = [{'node_i': e['node_i'], 'node_j': e['node_j'], 'E': e['E'], 'nu': e['nu'], 'A': e['A'], 'I_y': e['I_y'], 'I_z': e['I_z'], 'J': e['J'], 'local_z': np.dot(R_mat, e.get('local_z', np.array([0, 0, 1])))} for e in elements]
    u_global_rot = np.zeros_like(u_global)
    for i in range(node_coords.shape[0]):
        u_global_rot[i * 6:i * 6 + 3] = np.dot(R_mat, u_global[i * 6:i * 6 + 3])
        u_global_rot[i * 6 + 3:i * 6 + 6] = np.dot(R_mat, u_global[i * 6 + 3:i * 6 + 6])
    K_g = fcn(node_coords, elements, u_global)
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    T = np.block([[R_mat if i % 6 < 3 else R_mat for i in range(j * 6, (j + 1) * 6)] for j in range(node_coords.shape[0])])
    assert np.allclose(K_g_rot, np.dot(T, np.dot(K_g, T.T)))