def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness for a simple 3-node, 2-element chain:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    L = 2.0
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    E = 210000000000.0
    nu = 0.29
    A = 0.005
    I_y = 8.33e-06
    I_z = 8.33e-06
    J = 1.67e-05
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)

    def make_element(i, j, lz):
        return {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': lz, 'nodes': (i, j), 'node_ids': (i, j), 'conn': (i, j), 'i': i, 'j': j, 'n1': i, 'n2': j, 'node_i': i, 'node_j': j}
    elements = [make_element(0, 1, local_z), make_element(1, 2, local_z)]
    u0 = np.zeros(dof)
    K0 = fcn(node_coords, elements, u0)
    assert K0.shape == (dof, dof)
    assert np.allclose(K0, 0.0, atol=1e-12, rtol=0.0)
    rng = np.random.default_rng(12345)
    u = rng.standard_normal(dof)
    K = fcn(node_coords, elements, u)
    assert not np.allclose(K, 0.0, atol=1e-12)
    assert np.allclose(K, K.T, rtol=1e-09, atol=1e-10)
    alpha = 3.0
    K_alpha = fcn(node_coords, elements, alpha * u)
    assert np.allclose(K_alpha, alpha * K, rtol=1e-09, atol=1e-10)
    u1 = rng.standard_normal(dof)
    u2 = rng.standard_normal(dof)
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K12 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K12, K1 + K2, rtol=1e-09, atol=1e-10)
    elements_rev = list(reversed(elements))
    K_rev = fcn(node_coords, elements_rev, u)
    assert np.allclose(K_rev, K, rtol=1e-12, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity: rotating geometry, local axes, and displacement field
    with a global rotation R yields K_g^rot ≈ T K_g T^T, where T is block-diagonal
    with per-node blocks diag(R, R).
    """
    L = 1.7
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    E = 200000000000.0
    nu = 0.3
    A = 0.0042
    I_y = 7.8e-06
    I_z = 6.9e-06
    J = 1.1e-05
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)

    def make_element(i, j, lz):
        return {'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': lz, 'nodes': (i, j), 'node_ids': (i, j), 'conn': (i, j), 'i': i, 'j': j, 'n1': i, 'n2': j, 'node_i': i, 'node_j': j}
    elements = [make_element(0, 1, local_z), make_element(1, 2, local_z)]
    rng = np.random.default_rng(2024)
    u = rng.standard_normal(dof)
    K = fcn(node_coords, elements, u)

    def rotation_matrix(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        (ax, ay, az) = axis
        Kx = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], dtype=float)
        I3 = np.eye(3)
        return I3 + np.sin(angle) * Kx + (1.0 - np.cos(angle)) * (Kx @ Kx)
    axis = np.array([0.35, -0.2, 0.91], dtype=float)
    angle = 0.63
    R = rotation_matrix(axis, angle)
    T = np.zeros((dof, dof), dtype=float)
    for a in range(n_nodes):
        i0 = 6 * a
        T[i0:i0 + 3, i0:i0 + 3] = R
        T[i0 + 3:i0 + 6, i0 + 3:i0 + 6] = R
    node_coords_rot = node_coords @ R.T
    local_z_rot = (R @ local_z).astype(float)
    elements_rot = [make_element(0, 1, local_z_rot), make_element(1, 2, local_z_rot)]
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_pred = T @ K @ T.T
    assert np.allclose(K_rot, K_pred, rtol=5e-08, atol=1e-09)