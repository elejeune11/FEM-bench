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
    elem_props = dict(E=210000000000.0, nu=0.3, A=0.01, I_y=8e-06, I_z=6e-06, J=1.2e-05)
    elements = [{**elem_props, 'nodes': (0, 1)}, {**elem_props, 'nodes': (1, 2)}]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u0 = np.zeros(ndof, dtype=float)
    K0 = fcn(node_coords, elements, u0)
    assert K0.shape == (ndof, ndof)
    assert np.allclose(K0, 0.0, atol=1e-12)
    u1 = np.array([0.001, -0.0005, 0.0003, 0.01, -0.02, 0.015, -0.0002, 0.0004, -0.0001, -0.005, 0.01, -0.02, 0.0007, -0.0003, 0.0002, 0.02, -0.01, 0.005], dtype=float)
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, rtol=1e-09, atol=1e-10)
    alpha = 3.0
    K_alpha = fcn(node_coords, elements, alpha * u1)
    assert np.allclose(K_alpha, alpha * K1, rtol=1e-09, atol=1e-10)
    u2 = np.array([-0.0006, 0.0011, 0.0009, -0.015, 0.005, -0.007, 0.0003, -0.0008, 0.0002, 0.004, -0.006, 0.003, -0.0005, 0.0002, -0.0004, -0.009, 0.007, -0.002], dtype=float)
    K2 = fcn(node_coords, elements, u2)
    K12 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K12, K1 + K2, rtol=1e-08, atol=1e-10)
    elements_reversed = [{**elem_props, 'nodes': (1, 2)}, {**elem_props, 'nodes': (0, 1)}]
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
    L = 1.5
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    elem_props = dict(E=200000000000.0, nu=0.29, A=0.012, I_y=9e-06, I_z=7.5e-06, J=1.5e-05)
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [{**elem_props, 'nodes': (0, 1), 'local_z': local_z.copy()}, {**elem_props, 'nodes': (1, 2), 'local_z': local_z.copy()}]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u = np.array([0.0008, -0.0002, 0.0005, 0.012, -0.009, 0.004, -0.0004, 0.0007, -0.0001, -0.006, 0.003, -0.011, 0.0003, -0.0005, 0.0002, 0.008, -0.004, 0.006], dtype=float)

    def rodrigues(axis, theta):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        (x, y, z) = axis
        K = np.array([[0, -z, y], [z, 0, -x], [-y, x, 0]], dtype=float)
        I = np.eye(3)
        R = I + np.sin(theta) * K + (1 - np.cos(theta)) * (K @ K)
        return R
    R = rodrigues([1.0, 2.0, 3.0], 0.7)
    T_blocks = []
    for _ in range(n_nodes):
        T_blocks.append(np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]]))
    T = np.block([[T_blocks[i] if i == j else np.zeros((6, 6)) for j in range(n_nodes)] for i in range(n_nodes)])
    K = fcn(node_coords, elements, u)
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for e in elements:
        z_rot = np.asarray(e['local_z']) @ R.T
        elements_rot.append({**e, 'local_z': z_rot})
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_pred = T @ K @ T.T
    assert K_rot.shape == K_pred.shape == (ndof, ndof)
    assert np.allclose(K_rot, K_pred, rtol=1e-07, atol=1e-08)