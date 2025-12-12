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
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}]
    u_global = np.zeros(18)
    K_g = fcn(node_coords, elements, u_global)
    assert np.allclose(K_g, np.zeros_like(K_g))
    u_global = np.random.rand(18)
    K_g = fcn(node_coords, elements, u_global)
    assert np.allclose(K_g, K_g.T)
    u_global_1 = np.random.rand(18)
    u_global_2 = 2 * u_global_1
    K_g_1 = fcn(node_coords, elements, u_global_1)
    K_g_2 = fcn(node_coords, elements, u_global_2)
    assert np.allclose(K_g_2, 2 * K_g_1)
    u_global_1 = np.random.rand(18)
    u_global_2 = np.random.rand(18)
    K_g_1 = fcn(node_coords, elements, u_global_1)
    K_g_2 = fcn(node_coords, elements, u_global_2)
    K_g_sum = fcn(node_coords, elements, u_global_1 + u_global_2)
    assert np.allclose(K_g_sum, K_g_1 + K_g_2)
    elements_reversed = [{'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}]
    u_global = np.random.rand(18)
    K_g_original = fcn(node_coords, elements, u_global)
    K_g_reversed = fcn(node_coords, elements_reversed, u_global)
    assert np.allclose(K_g_original, K_g_reversed)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}]
    u_global = np.random.rand(18)
    K_g = fcn(node_coords, elements, u_global)
    rot = R.from_euler('xyz', [np.pi / 4, 0, 0])
    R_mat = rot.as_matrix()
    node_coords_rot = np.dot(node_coords, R_mat.T)
    elements_rot = []
    for elem in elements:
        if 'local_z' in elem:
            elem_rot = elem.copy()
            elem_rot['local_z'] = np.dot(np.array(elem['local_z']), R_mat.T)
            elements_rot.append(elem_rot)
        else:
            elements_rot.append(elem)
    T = np.block([[R_mat] * 3, np.zeros((3, 3))] * 3)
    u_global_rot = np.dot(T, u_global)
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    T = np.tile(np.eye(6), (3, 1)).reshape(18, 18)
    for i in range(3):
        T[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6] = R_mat
    assert np.allclose(K_g_rot, np.dot(np.dot(T, K_g), T.T))