def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness for a 3-node, 2-element chain:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8e-06
    I_z = 6e-06
    J = 1.2e-05
    lz = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': lz}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': lz}]
    ndof = 6 * n_nodes
    u_zero = np.zeros(ndof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (ndof, ndof)
    assert np.allclose(K_zero, 0.0, atol=1e-12)
    u1 = np.concatenate([np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]), np.array([0.001, 0.003, -0.002, 0.01, -0.02, 0.03]), np.array([0.0, 0.001, 0.002, -0.01, 0.02, -0.04])])
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, rtol=1e-09, atol=1e-10)
    alpha = 2.75
    K1_scaled = fcn(node_coords, elements, alpha * u1)
    assert np.allclose(K1_scaled, alpha * K1, rtol=1e-08, atol=1e-10)
    u2 = np.concatenate([np.array([0.002, -0.001, 0.0025, -0.005, 0.007, 0.0015]), np.array([-0.0005, 0.0015, 0.0, 0.02, 0.01, 0.0]), np.array([0.0, -0.0025, 0.001, -0.03, 0.0, 0.015])])
    K_u1 = fcn(node_coords, elements, u1)
    K_u2 = fcn(node_coords, elements, u2)
    K_u12 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_u12, K_u1 + K_u2, rtol=1e-08, atol=1e-10)
    elements_reversed = [elements[1].copy(), elements[0].copy()]
    K1_rev = fcn(node_coords, elements_reversed, u1)
    assert np.allclose(K1_rev, K1, rtol=1e-10, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity: rotating geometry, local axes, and displacement field by R
    yields K_g^rot ≈ T K_g T^T, where T is block-diagonal per node with diag(R, R).
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.5, 0.0, 0.0], [3.0, 0.0, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8e-06
    I_z = 6e-06
    J = 1.2e-05
    lz = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': lz}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': lz}]
    rng = np.random.default_rng(12345)
    u = rng.normal(scale=0.01, size=ndof)
    K = fcn(node_coords, elements, u)
    axis = np.array([1.0, -2.0, 1.5], dtype=float)
    axis = axis / np.linalg.norm(axis)
    theta = 0.7
    Kx = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    R = np.eye(3) * np.cos(theta) + np.sin(theta) * Kx + (1.0 - np.cos(theta)) * np.outer(axis, axis)
    T = np.zeros((ndof, ndof))
    for i in range(n_nodes):
        i6 = 6 * i
        T[i6:i6 + 3, i6:i6 + 3] = R
        T[i6 + 3:i6 + 6, i6 + 3:i6 + 6] = R
    node_coords_rot = node_coords @ R.T
    lz_rot = R @ lz
    elements_rot = [{'node_i': e['node_i'], 'node_j': e['node_j'], 'E': e['E'], 'nu': e['nu'], 'A': e['A'], 'I_y': e['I_y'], 'I_z': e['I_z'], 'J': e['J'], 'local_z': lz_rot} for e in elements]
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_map = T @ K @ T.T
    assert K_rot.shape == (ndof, ndof)
    assert np.allclose(K_rot, K_map, rtol=1e-08, atol=1e-10)