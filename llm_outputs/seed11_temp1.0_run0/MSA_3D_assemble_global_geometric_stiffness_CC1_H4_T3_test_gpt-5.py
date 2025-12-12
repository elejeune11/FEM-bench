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
    Iy = 1e-06
    Iz = 1.2e-06
    J = 2e-06
    ez = [0.0, 0.0, 1.0]
    elements = [{'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': ez, 'nodes': (0, 1), 'node_i': 0, 'node_j': 1}, {'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': ez, 'nodes': (1, 2), 'node_i': 1, 'node_j': 2}]
    u0 = np.zeros(6 * n_nodes, dtype=float)
    K0 = fcn(node_coords, elements, u0)
    assert np.allclose(K0, 0.0, atol=1e-12, rtol=0.0)
    u1 = np.zeros_like(u0)
    u1[6 * 2 + 0] = 0.0001
    u1[6 * 1 + 3] = 0.0002
    u2 = np.zeros_like(u0)
    u2[6 * 1 + 1] = 5e-05
    u2[6 * 2 + 1] = 0.0001
    u2[6 * 2 + 5] = -0.0003
    u = u1 + u2
    K = fcn(node_coords, elements, u)
    assert np.allclose(K, K.T, rtol=1e-08, atol=1e-10)
    s = 3.2
    Ks = fcn(node_coords, elements, s * u)
    assert np.allclose(Ks, s * K, rtol=1e-08, atol=1e-10)
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    assert np.allclose(K, K1 + K2, rtol=1e-08, atol=1e-10)
    elements_reversed = list(reversed(elements))
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
    n_nodes = node_coords.shape[0]
    E = 200000000000.0
    nu = 0.29
    A = 0.012
    Iy = 9e-07
    Iz = 1.1e-06
    J = 1.8e-06
    ez = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [{'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': ez, 'nodes': (0, 1), 'node_i': 0, 'node_j': 1}, {'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': ez, 'nodes': (1, 2), 'node_i': 1, 'node_j': 2}]
    u = np.zeros(6 * n_nodes, dtype=float)
    u[6 * 1 + 0] = 8e-05
    u[6 * 1 + 2] = -4e-05
    u[6 * 1 + 4] = 0.00015
    u[6 * 2 + 1] = 0.00012
    u[6 * 2 + 3] = -0.0002
    u[6 * 2 + 5] = 0.00025
    K = fcn(node_coords, elements, u)
    theta = np.deg2rad(37.0)
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    T = np.zeros((6 * n_nodes, 6 * n_nodes), dtype=float)
    for a in range(n_nodes):
        T[6 * a:6 * a + 3, 6 * a:6 * a + 3] = R
        T[6 * a + 3:6 * a + 6, 6 * a + 3:6 * a + 6] = R
    node_coords_rot = node_coords @ R.T
    u_rot = T @ u
    ez_rot = R @ ez
    elements_rot = []
    for e in elements:
        e_rot = dict(e)
        e_rot['local_z'] = ez_rot
        elements_rot.append(e_rot)
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    TKTT = T @ K @ T.T
    assert np.allclose(K_rot, TKTT, rtol=5e-07, atol=1e-09)