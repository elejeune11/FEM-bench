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
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    n_nodes = node_coords.shape[0]
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}]
    u_zero = np.zeros(6 * n_nodes)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0), 'Zero displacement should produce zero geometric stiffness matrix'
    u_random = np.random.rand(6 * n_nodes)
    K_random = fcn(node_coords, elements, u_random)
    assert np.allclose(K_random, K_random.T), 'Geometric stiffness matrix should be symmetric'
    scale = 2.5
    K_scaled = fcn(node_coords, elements, scale * u_random)
    assert np.allclose(K_scaled, scale * K_random), 'Scaling displacements should scale K_g linearly'
    u1 = np.random.rand(6 * n_nodes)
    u2 = np.random.rand(6 * n_nodes)
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2), 'Superposition should hold for independent displacement states'
    elements_reversed = [elements[1], elements[0]]
    K_reversed = fcn(node_coords, elements_reversed, u_random)
    assert np.allclose(K_random, K_reversed), 'Element order should not affect the assembled matrix'

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    n_nodes = node_coords.shape[0]
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}]
    u_original = np.random.rand(6 * n_nodes)
    K_original = fcn(node_coords, elements, u_original)
    rotation = R.random()
    R_matrix = rotation.as_matrix()
    node_coords_rot = (R_matrix @ node_coords.T).T
    elements_rot = []
    for ele in elements:
        ele_copy = ele.copy()
        elements_rot.append(ele_copy)
    u_rot = np.zeros_like(u_original)
    for i in range(n_nodes):
        u_block = u_original[6 * i:6 * i + 6]
        u_trans = R_matrix @ u_block[0:3]
        u_rot_block = R_matrix @ u_block[3:6]
        u_rot[6 * i:6 * i + 3] = u_trans
        u_rot[6 * i + 3:6 * i + 6] = u_rot_block
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    T = np.kron(np.eye(n_nodes), np.kron(np.eye(2), R_matrix))
    K_transformed = T @ K_original @ T.T
    assert np.allclose(K_rot, K_transformed, atol=1e-06), 'Frame objectivity violated: K_rot != T K T^T'