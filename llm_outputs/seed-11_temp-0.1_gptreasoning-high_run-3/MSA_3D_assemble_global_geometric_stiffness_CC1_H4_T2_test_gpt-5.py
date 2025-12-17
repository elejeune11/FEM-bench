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
    common = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.5e-06, 'I_z': 8.5e-06, 'J': 1.7e-05, 'local_z': np.array([0.0, 0.0, 1.0], dtype=float)}
    elements = [{'node_i': 0, 'node_j': 1, **common}, {'node_i': 1, 'node_j': 2, **common}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u_zero = np.zeros(dof, dtype=float)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (dof, dof)
    assert np.allclose(K_zero, np.zeros_like(K_zero), rtol=0, atol=1e-12)
    u = np.array([0.0, 0.002, -0.001, 0.001, -0.0005, 0.0007, 0.001, -0.001, 0.002, -0.0008, 0.0006, -0.0004, -0.001, 0.001, -0.002, 0.0009, -0.0003, 0.0002], dtype=float)
    K = fcn(node_coords, elements, u)
    assert np.allclose(K, K.T, rtol=1e-10, atol=1e-12)
    s = 2.3
    K_scaled = fcn(node_coords, elements, s * u)
    assert np.allclose(K_scaled, s * K, rtol=1e-10, atol=1e-09)
    u_a = np.array([0.0005, -0.0007, 0.0009, 0.0003, -0.0002, 0.0004, -0.0004, 0.0006, -0.0008, 0.0002, 0.0001, -0.0003, 0.0007, -0.0005, 0.0004, -0.0006, 0.0002, -0.0001], dtype=float)
    u_b = np.array([-0.0002, 0.0003, -0.0001, 0.0005, 0.0004, -0.0002, 0.0006, -0.0001, 0.0002, -0.0004, 0.0003, 0.0001, -0.0003, -0.0002, 0.0005, 0.0001, -0.0004, 0.0006], dtype=float)
    K_super = fcn(node_coords, elements, u_a + u_b)
    K_sum = fcn(node_coords, elements, u_a) + fcn(node_coords, elements, u_b)
    assert np.allclose(K_super, K_sum, rtol=1e-10, atol=1e-09)
    elements_reversed = [{'node_i': 1, 'node_j': 2, **common}, {'node_i': 0, 'node_j': 1, **common}]
    K_rev = fcn(node_coords, elements_reversed, u)
    assert np.allclose(K, K_rev, rtol=1e-12, atol=1e-12)

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
    common = {'E': 200000000000.0, 'nu': 0.29, 'A': 0.012, 'I_y': 9.1e-06, 'I_z': 7.8e-06, 'J': 1.5e-05, 'local_z': np.array([0.0, 0.0, 1.0], dtype=float)}
    elements = [{'node_i': 0, 'node_j': 1, **common}, {'node_i': 1, 'node_j': 2, **common}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u = np.array([0.0006, -0.0011, 0.0009, 0.0004, -0.0003, 0.0002, -0.0007, 0.0012, -0.0004, 0.0006, 0.0001, -0.0005, 0.001, -0.0008, 0.0003, -0.0002, 0.0005, 0.0007], dtype=float)
    K = fcn(node_coords, elements, u)

    def Rz(a):
        ca, sa = (np.cos(a), np.sin(a))
        return np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]], dtype=float)

    def Ry(b):
        cb, sb = (np.cos(b), np.sin(b))
        return np.array([[cb, 0.0, sb], [0.0, 1.0, 0.0], [-sb, 0.0, cb]], dtype=float)

    def Rx(c):
        cc, sc = (np.cos(c), np.sin(c))
        return np.array([[1.0, 0.0, 0.0], [0.0, cc, -sc], [0.0, sc, cc]], dtype=float)
    R = Rz(0.7) @ Ry(-0.35) @ Rx(0.45)
    T = np.zeros((dof, dof), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for e in elements:
        e_rot = {'node_i': e['node_i'], 'node_j': e['node_j'], 'E': e['E'], 'nu': e['nu'], 'A': e['A'], 'I_y': e['I_y'], 'I_z': e['I_z'], 'J': e['J'], 'local_z': (R @ np.asarray(e['local_z'], dtype=float)).astype(float)}
        elements_rot.append(e_rot)
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_expected = T @ K @ T.T
    assert np.allclose(K_rot, K_expected, rtol=1e-08, atol=1e-10)