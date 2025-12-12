def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    n_nodes = node_coords.shape[0]
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}]
    ndof = 6 * n_nodes
    u_zero = np.zeros(ndof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0), 'Zero displacement should produce zero stiffness matrix'
    u_nonzero = np.random.rand(ndof)
    K = fcn(node_coords, elements, u_nonzero)
    assert np.allclose(K, K.T), 'Geometric stiffness matrix should be symmetric'
    scale = 2.5
    K_scaled = fcn(node_coords, elements, scale * u_nonzero)
    assert np.allclose(K_scaled, scale * K), 'Scaling displacements should scale K_g linearly'
    u1 = np.random.rand(ndof)
    u2 = np.random.rand(ndof)
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2), 'Superposition should hold for geometric stiffness'
    elements_reversed = [elements[1], elements[0]]
    K_reversed = fcn(node_coords, elements_reversed, u_nonzero)
    assert np.allclose(K, K_reversed), 'Element order should not affect assembled matrix'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    n_nodes = node_coords.shape[0]
    elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}]
    ndof = 6 * n_nodes
    u_original = np.random.rand(ndof)
    K_original = fcn(node_coords, elements, u_original)
    rot_vec = np.random.rand(3) * 2 * np.pi
    rotation = R.from_rotvec(rot_vec)
    R_matrix = rotation.as_matrix()
    node_coords_rot = (R_matrix @ node_coords.T).T
    elements_rot = []
    for elem in elements:
        new_elem = elem.copy()
        elements_rot.append(new_elem)
    u_rot = np.zeros_like(u_original)
    for i in range(n_nodes):
        u_node = u_original[6 * i:6 * i + 3]
        theta_node = u_original[6 * i + 3:6 * i + 6]
        u_rot[6 * i:6 * i + 3] = R_matrix @ u_node
        u_rot[6 * i + 3:6 * i + 6] = R_matrix @ theta_node
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    T = np.kron(np.eye(n_nodes), np.kron(np.eye(2), R_matrix))
    K_transformed = T @ K_original @ T.T
    assert np.allclose(K_rot, K_transformed, atol=1e-08), 'Frame objectivity violated under global rotation'