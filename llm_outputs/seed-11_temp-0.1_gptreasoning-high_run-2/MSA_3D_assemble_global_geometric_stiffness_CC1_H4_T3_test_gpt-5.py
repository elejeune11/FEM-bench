def test_multi_element_core_correctness_assembly(fcn):
    """
    Verify basic correctness of assemble_global_geometric_stiffness_3D_beam for a simple 3-node, 2-element chain.
    Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    L = 2.0
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 8.333e-06
    Iz = 6.667e-06
    J = 1.234e-05
    ref_z = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': ref_z}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': ref_z}]
    u0 = np.zeros(ndof)
    K0 = fcn(node_coords, elements, u0)
    assert isinstance(K0, np.ndarray)
    assert K0.shape == (ndof, ndof)
    assert np.allclose(K0, 0.0, atol=1e-12)
    rng = np.random.default_rng(12345)
    u = rng.normal(size=ndof)
    K = fcn(node_coords, elements, u)
    assert K.shape == (ndof, ndof)
    assert np.linalg.norm(K) > 1e-12
    assert np.allclose(K, K.T, rtol=1e-09, atol=1e-12)
    alpha = -3.7
    K_alpha = fcn(node_coords, elements, alpha * u)
    assert np.allclose(K_alpha, alpha * K, rtol=1e-08, atol=1e-10)
    u1 = rng.normal(size=ndof)
    u2 = rng.normal(size=ndof)
    K_u1 = fcn(node_coords, elements, u1)
    K_u2 = fcn(node_coords, elements, u2)
    K_u12 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_u12, K_u1 + K_u2, rtol=1e-08, atol=1e-10)
    elements_reversed = [elements[1], elements[0]]
    K_rev = fcn(node_coords, elements_reversed, u)
    assert np.allclose(K, K_rev, rtol=1e-12, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by a global rotation R
    should produce a geometric stiffness matrix K_g^rot ≈ T K_g T^T, where T is block-diagonal with
    per-node blocks diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    L = 2.0
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 8.333e-06
    Iz = 6.667e-06
    J = 1.234e-05
    ref_z = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': ref_z}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': ref_z}]
    u = np.zeros(ndof)
    u[0] = 0.0
    u[6] = 0.001
    u[12] = 0.002

    def rodrigues(axis, theta):
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        ax, ay, az = a
        K = np.array([[0.0, -az, ay], [az, 0.0, -ax], [-ay, ax, 0.0]], dtype=float)
        I = np.eye(3)
        R = I * np.cos(theta) + (1 - np.cos(theta)) * np.outer(a, a) + np.sin(theta) * K
        return R
    axis = [1.0, 2.0, -1.0]
    theta = 0.731
    R = rodrigues(axis, theta)
    T = np.zeros((ndof, ndof), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    K = fcn(node_coords, elements, u)
    assert np.linalg.norm(K) > 0.0
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for e in elements:
        z_local = np.asarray(e['local_z'], dtype=float)
        z_rot = z_local @ R.T
        e_rot = dict(e)
        e_rot['local_z'] = z_rot / np.linalg.norm(z_rot)
        elements_rot.append(e_rot)
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_expected = T @ K @ T.T
    assert np.allclose(K_rot, K_expected, rtol=1e-07, atol=1e-09)