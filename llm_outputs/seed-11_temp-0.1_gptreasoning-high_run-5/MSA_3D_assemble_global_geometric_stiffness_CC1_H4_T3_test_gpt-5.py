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
    L = 2.0
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 2e-06
    I_z = 1.5e-06
    J = 3.1e-06
    local_z = [0.0, 0.0, 1.0]
    elements = [dict(node_i=0, node_j=1, E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=local_z), dict(node_i=1, node_j=2, E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=local_z)]
    ndof = 6 * n_nodes
    u0 = np.zeros(ndof)
    K0 = fcn(node_coords, elements, u0)
    assert K0.shape == (ndof, ndof)
    assert np.allclose(K0, np.zeros_like(K0), atol=1e-12, rtol=0.0)
    u1 = np.zeros(ndof)
    u1[6 * 2 + 0] = 0.001
    u1[6 * 1 + 1] = 0.002
    u1[6 * 1 + 5] = 0.005
    u1[6 * 0 + 2] = -0.001
    u1[6 * 0 + 3] = 0.004
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, rtol=1e-10, atol=1e-12)
    assert np.linalg.norm(K1) > 0.0
    alpha = 3.0
    K_alpha = fcn(node_coords, elements, alpha * u1)
    assert np.allclose(K_alpha, alpha * K1, rtol=1e-10, atol=1e-12)
    u2 = np.zeros(ndof)
    u2[6 * 0 + 0] = -0.002
    u2[6 * 2 + 2] = 0.0015
    u2[6 * 2 + 4] = 0.002
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2, rtol=1e-10, atol=1e-12)
    elements_reordered = list(reversed(elements))
    K1_reordered = fcn(node_coords, elements_reordered, u1)
    assert np.allclose(K1_reordered, K1, rtol=1e-12, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """

    def rot_x(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[1, 0, 0], [0, ca, -sa], [0, sa, ca]], dtype=float)

    def rot_y(b):
        cb, sb = (np.cos(b), np.sin(b))
        return np.array([[cb, 0, sb], [0, 1, 0], [-sb, 0, cb]], dtype=float)

    def rot_z(g):
        cg, sg = (np.cos(g), np.sin(g))
        return np.array([[cg, -sg, 0], [sg, cg, 0], [0, 0, 1]], dtype=float)
    alpha, beta, gamma = (0.31, -0.47, 0.52)
    R = rot_z(gamma) @ rot_y(beta) @ rot_x(alpha)
    L = 1.5
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    E = 200000000000.0
    nu = 0.29
    A = 0.012
    I_y = 1.8e-06
    I_z = 2.2e-06
    J = 3.5e-06
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [dict(node_i=0, node_j=1, E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=local_z.copy()), dict(node_i=1, node_j=2, E=E, nu=nu, A=A, I_y=I_y, I_z=I_z, J=J, local_z=local_z.copy())]
    u = np.zeros(ndof)
    u[0:3] = [0.001, -0.002, 0.0015]
    u[3:6] = [0.01, -0.013, 0.007]
    base1 = 6
    u[base1 + 0:base1 + 3] = [-0.0005, 0.0008, -0.0012]
    u[base1 + 3:base1 + 6] = [0.003, 0.02, -0.004]
    base2 = 12
    u[base2 + 0:base2 + 3] = [0.002, 0.001, -0.001]
    u[base2 + 3:base2 + 6] = [-0.006, 0.005, 0.009]
    K = fcn(node_coords, elements, u)
    block = np.zeros((6, 6), dtype=float)
    block[:3, :3] = R
    block[3:, 3:] = R
    T = np.kron(np.eye(n_nodes), block)
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for e in elements:
        e_rot = dict(e)
        z_rot = np.array(e['local_z'], dtype=float) @ R.T
        e_rot['local_z'] = z_rot
        elements_rot.append(e_rot)
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_pred = T @ K @ T.T
    assert K_rot.shape == K.shape == (ndof, ndof)
    assert np.allclose(K_rot, K_pred, rtol=1e-08, atol=1e-10)