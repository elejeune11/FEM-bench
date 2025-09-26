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
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.5, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.5, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u_zero = np.zeros(ndof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert isinstance(K_zero, np.ndarray)
    assert K_zero.shape == (ndof, ndof)
    assert np.allclose(K_zero, 0.0, atol=1e-12)
    rng = np.random.default_rng(12345)
    u1 = rng.normal(size=ndof)
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, rtol=1e-09, atol=1e-09)
    alpha = 0.7
    K_alpha = fcn(node_coords, elements, alpha * u1)
    assert np.allclose(K_alpha, alpha * K1, rtol=1e-08, atol=1e-10)
    u2 = rng.normal(size=ndof)
    K2 = fcn(node_coords, elements, u2)
    K12 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K12, K1 + K2, rtol=1e-08, atol=1e-10)
    elements_reversed = list(reversed(elements))
    K1_rev = fcn(node_coords, elements_reversed, u1)
    assert np.allclose(K1_rev, K1, rtol=1e-10, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    base_local_z = np.array([0.0, 0.0, 1.0])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.5, 'local_z': base_local_z.tolist()}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.5, 'local_z': base_local_z.tolist()}]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    rng = np.random.default_rng(2021)
    u = rng.normal(size=ndof)
    K = fcn(node_coords, elements, u)
    axis = np.array([0.3, -0.5, 0.8], dtype=float)
    axis /= np.linalg.norm(axis)
    theta = 0.7
    Kx = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    R = np.eye(3) + np.sin(theta) * Kx + (1.0 - np.cos(theta)) * (Kx @ Kx)
    node_coords_rot = node_coords @ R.T
    z_rot = (R @ base_local_z).tolist()
    elements_rot = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.5, 'local_z': z_rot}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.5, 'local_z': z_rot}]
    T = np.zeros((ndof, ndof))
    for a in range(n_nodes):
        i = 6 * a
        T[i:i + 3, i:i + 3] = R
        T[i + 3:i + 6, i + 3:i + 6] = R
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_pred = T @ K @ T.T
    assert np.allclose(K_rot, K_pred, rtol=1e-07, atol=1e-09)