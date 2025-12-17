def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness for a simple 3-node, 2-element chain:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric for a non-zero displacement state,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    L = 1.5
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    props = dict(E=210000000000.0, nu=0.3, A=0.008, I_y=3e-06, I_z=2e-06, J=5e-06)
    local_z = np.array([0.0, 0.0, 1.0])
    elements = [dict(node_i=0, node_j=1, local_z=local_z, **props), dict(node_i=1, node_j=2, local_z=local_z, **props)]
    dof = 6 * n_nodes
    u0 = np.zeros(dof)
    K0 = fcn(node_coords=node_coords, elements=elements, u_global=u0)
    assert K0.shape == (dof, dof)
    assert np.allclose(K0, 0.0, atol=1e-12)
    rng = np.random.default_rng(12345)
    u = 0.0001 * rng.standard_normal(dof)
    K = fcn(node_coords=node_coords, elements=elements, u_global=u)
    sym_err = np.linalg.norm(K - K.T, ord='fro')
    base = np.linalg.norm(K, ord='fro')
    assert sym_err <= 1e-10 * (1.0 + base)
    alpha = 1.234
    K_alpha = fcn(node_coords=node_coords, elements=elements, u_global=alpha * u)
    assert np.allclose(K_alpha, alpha * K, rtol=1e-08, atol=1e-12 * (1.0 + base))
    u1 = 0.0001 * rng.standard_normal(dof)
    u2 = 0.0001 * rng.standard_normal(dof)
    K1 = fcn(node_coords=node_coords, elements=elements, u_global=u1)
    K2 = fcn(node_coords=node_coords, elements=elements, u_global=u2)
    K12 = fcn(node_coords=node_coords, elements=elements, u_global=u1 + u2)
    norm12 = np.linalg.norm(K12, ord='fro')
    assert np.allclose(K12, K1 + K2, rtol=1e-08, atol=1e-12 * (1.0 + norm12))
    elements_rev = list(reversed(elements))
    K_rev = fcn(node_coords=node_coords, elements=elements_rev, u_global=u)
    assert np.allclose(K_rev, K, rtol=1e-10, atol=1e-12 * (1.0 + base))

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity:
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce K_g^rot ≈ T K_g T^T, where T is block-diagonal
    with per-node blocks diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    L = 2.0
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    props = dict(E=210000000000.0, nu=0.3, A=0.01, I_y=4e-06, I_z=5e-06, J=7e-06)
    local_z = np.array([0.0, 0.0, 1.0])
    elements = [dict(node_i=0, node_j=1, local_z=local_z, **props), dict(node_i=1, node_j=2, local_z=local_z, **props)]
    rng = np.random.default_rng(202)
    u = 5e-05 * rng.standard_normal(dof)
    K = fcn(node_coords=node_coords, elements=elements, u_global=u)

    def rotation_matrix(axis, theta):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        ax, ay, az = axis
        Kx = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]])
        I3 = np.eye(3)
        return I3 + np.sin(theta) * Kx + (1.0 - np.cos(theta)) * (Kx @ Kx)
    axis = np.array([0.3, -0.7, 0.65])
    theta = 0.73
    R = rotation_matrix(axis, theta)
    node_coords_rot = (R @ node_coords.T).T
    local_z_rot = (R @ local_z.reshape(3, 1)).ravel()
    elements_rot = [dict(node_i=e['node_i'], node_j=e['node_j'], local_z=local_z_rot, **{k: e[k] for k in ['E', 'nu', 'A', 'I_y', 'I_z', 'J']}) for e in elements]
    T_node = np.zeros((6, 6))
    T_node[:3, :3] = R
    T_node[3:, 3:] = R
    T = np.zeros((dof, dof))
    for i in range(n_nodes):
        T[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6] = T_node
    u_rot = T @ u
    K_rot = fcn(node_coords=node_coords_rot, elements=elements_rot, u_global=u_rot)
    TKTT = T @ K @ T.T
    scale = 1.0 + max(np.linalg.norm(K, ord='fro'), np.linalg.norm(K_rot, ord='fro'))
    assert np.allclose(K_rot, TKTT, rtol=5e-08, atol=1e-12 * scale)