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
    I_y = 2e-05
    I_z = 1e-05
    J = 1.5e-05
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}]
    u0 = np.zeros(6 * n_nodes, dtype=float)
    K0 = fcn(node_coords, elements, u0)
    assert K0.shape == (6 * n_nodes, 6 * n_nodes)
    assert np.allclose(K0, np.zeros_like(K0), rtol=0, atol=1e-12)
    u1 = np.zeros_like(u0)
    u1[0:6] = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    u1[6:12] = np.array([0.001, 0.0002, 0.0, 0.0, 0.0, 0.0005])
    u1[12:18] = np.array([0.002, 0.0, -0.0001, 0.0, -0.0003, 0.0])
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, rtol=1e-08, atol=1e-10)
    scale = 2.7
    K1_scaled = fcn(node_coords, elements, scale * u1)
    assert np.allclose(K1_scaled, scale * K1, rtol=1e-08, atol=1e-10)
    u2 = np.zeros_like(u0)
    u2[0:6] = np.array([0.0, 0.0001, 0.0, 0.0002, 0.0, 0.0])
    u2[6:12] = np.array([0.0, 0.0, 0.0004, -0.0001, 0.00025, -5e-05])
    u2[12:18] = np.array([0.0, -0.0003, 0.0, 0.00011, 0.0, 0.0])
    K2 = fcn(node_coords, elements, u2)
    K12 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K12, K1 + K2, rtol=1e-08, atol=1e-10)
    elements_reversed = list(reversed(elements))
    K12_rev = fcn(node_coords, elements_reversed, u1 + u2)
    assert np.allclose(K12_rev, K12, rtol=1e-12, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """

    def rodrigues(axis, theta):
        a = np.asarray(axis, dtype=float)
        a = a / np.linalg.norm(a)
        ax = np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]], dtype=float)
        I = np.eye(3)
        return I + np.sin(theta) * ax + (1.0 - np.cos(theta)) * (ax @ ax)
    L = 1.7
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    E = 200000000000.0
    nu = 0.29
    A = 0.012
    I_y = 2.5e-05
    I_z = 1.2e-05
    J = 1.8e-05
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z}]
    u = np.zeros(6 * n_nodes, dtype=float)
    u[0:6] = np.array([0.001, 0.0005, -0.0003, 0.0002, -0.0001, 0.00015])
    u[6:12] = np.array([-0.0002, 0.0003, 0.0004, -0.0001, 0.00025, -5e-05])
    u[12:18] = np.array([0.0006, -0.00035, 0.0, 0.00018, 9e-05, -0.00011])
    K = fcn(node_coords, elements, u)
    axis = np.array([0.2, 0.9, -0.3], dtype=float)
    theta = 0.73
    R = rodrigues(axis, theta)
    block = np.zeros((6, 6), dtype=float)
    block[:3, :3] = R
    block[3:, 3:] = R
    T = np.kron(np.eye(n_nodes), block)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for e in elements:
        e_rot = dict(e)
        e_rot['local_z'] = R @ np.asarray(e['local_z'], dtype=float)
        elements_rot.append(e_rot)
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_expected = T @ K @ T.T
    assert np.allclose(K_rot, K_expected, rtol=1e-07, atol=1e-09)