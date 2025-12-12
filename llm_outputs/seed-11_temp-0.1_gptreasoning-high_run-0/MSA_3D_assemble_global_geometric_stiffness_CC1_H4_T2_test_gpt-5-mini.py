def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = [{'nodes': (0, 1), 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0.0, 0.0, 1.0])}, {'nodes': (1, 2), 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': np.array([0.0, 0.0, 1.0])}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u_zero = np.zeros(dof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (dof, dof)
    assert np.allclose(K_zero, np.zeros((dof, dof)), atol=1e-12, rtol=0.0)
    u = np.array([0.001, 0.002, -0.0005, 0.0001, -0.0002, 8e-05, -0.0005, 0.001, 0.002, -6e-05, 3e-05, -0.0001, 0.002, -0.001, 0.0005, 0.0002, -0.0001, 5e-05], dtype=float)
    K = fcn(node_coords, elements, u)
    assert np.allclose(K, K.T, atol=1e-10, rtol=1e-08)
    c = 2.5
    K_c = fcn(node_coords, elements, c * u)
    assert np.allclose(K_c, c * K, rtol=1e-06, atol=1e-10)
    u1 = u.copy()
    u1[6:] = 0.0
    u2 = u - u1
    K_u1 = fcn(node_coords, elements, u1)
    K_u2 = fcn(node_coords, elements, u2)
    K_u_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_u_sum, K_u1 + K_u2, rtol=1e-06, atol=1e-09)
    elements_rev = [elements[1], elements[0]]
    K_rev = fcn(node_coords, elements_rev, u)
    assert np.allclose(K_rev, K, rtol=1e-10, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    local_z = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [{'nodes': (0, 1), 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': local_z.copy()}, {'nodes': (1, 2), 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 1e-06, 'local_z': local_z.copy()}]
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u = np.array([0.001, 0.002, -0.0005, 0.0001, -0.0002, 8e-05, -0.0005, 0.001, 0.002, -6e-05, 3e-05, -0.0001, 0.002, -0.001, 0.0005, 0.0002, -0.0001, 5e-05], dtype=float)
    K = fcn(node_coords, elements, u)
    theta = 0.37
    c = np.cos(theta)
    s = np.sin(theta)
    R = np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]], dtype=float)
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for el in elements:
        el_rot = el.copy()
        el_rot['local_z'] = (R @ el['local_z']).astype(float)
        elements_rot.append(el_rot)
    block = np.zeros((6, 6), dtype=float)
    block[:3, :3] = R
    block[3:, 3:] = R
    T = np.kron(np.eye(n_nodes), block)
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_expected = T @ K @ T.T
    assert np.allclose(K_rot, K_expected, rtol=1e-06, atol=1e-08)