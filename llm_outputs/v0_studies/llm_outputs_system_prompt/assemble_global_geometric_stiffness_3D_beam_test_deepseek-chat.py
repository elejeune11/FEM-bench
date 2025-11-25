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
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-05, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-05, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    u_zero = np.zeros(6 * n_nodes)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0)
    u_test = np.random.rand(6 * n_nodes)
    K = fcn(node_coords, elements, u_test)
    assert np.allclose(K, K.T)
    scale = 2.5
    K_scaled = fcn(node_coords, elements, scale * u_test)
    assert np.allclose(K_scaled, scale * K)
    u1 = np.random.rand(6 * n_nodes)
    u2 = np.random.rand(6 * n_nodes)
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2)
    elements_reordered = [elements[1], elements[0]]
    K_reordered = fcn(node_coords, elements_reordered, u_test)
    assert np.allclose(K_reordered, K)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-05, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-05, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    u_global = np.random.rand(6 * n_nodes)
    K_original = fcn(node_coords, elements, u_global)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for ele in elements:
        ele_rot = ele.copy()
        if 'local_z' in ele:
            ele_rot['local_z'] = list(np.array(ele['local_z']) @ R.T)
        elements_rot.append(ele_rot)
    u_rotated = u_global.reshape(-1, 6)
    u_rotated[:, :3] = u_rotated[:, :3] @ R.T
    u_rotated[:, 3:] = u_rotated[:, 3:] @ R.T
    u_rotated = u_rotated.flatten()
    K_rotated = fcn(node_coords_rot, elements_rot, u_rotated)
    T = np.kron(np.eye(2 * n_nodes), R)
    K_transformed = T @ K_original @ T.T
    assert np.allclose(K_rotated, K_transformed, atol=1e-10)