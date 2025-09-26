def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}]
    n_nodes = node_coords.shape[0]
    u_zero = np.zeros(6 * n_nodes)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0), 'Zero displacement should produce zero matrix'
    assert np.allclose(K_zero, K_zero.T), 'Matrix should be symmetric'
    u_test = np.random.rand(6 * n_nodes)
    K1 = fcn(node_coords, elements, u_test)
    K2 = fcn(node_coords, elements, 2.0 * u_test)
    assert np.allclose(K2, 2.0 * K1, rtol=1e-10), 'Scaling should be linear'
    u1 = np.random.rand(6 * n_nodes)
    u2 = np.random.rand(6 * n_nodes)
    K_sum = fcn(node_coords, elements, u1 + u2)
    K1_plus_K2 = fcn(node_coords, elements, u1) + fcn(node_coords, elements, u2)
    assert np.allclose(K_sum, K1_plus_K2, rtol=1e-10), 'Superposition should hold'
    elements_reversed = list(reversed(elements))
    K_original = fcn(node_coords, elements, u_test)
    K_reversed = fcn(node_coords, elements_reversed, u_test)
    assert np.allclose(K_original, K_reversed), 'Element order should not affect result'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""

    def rotation_matrix(axis, angle):
        axis = axis / np.linalg.norm(axis)
        (c, s) = (np.cos(angle), np.sin(angle))
        return c * np.eye(3) + (1 - c) * np.outer(axis, axis) + s * np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    R = rotation_matrix([1, 1, 1], np.pi / 4)
    original_nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    rotated_nodes = (R @ original_nodes.T).T
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}]
    rotated_elements = []
    for ele in elements:
        rotated_ele = ele.copy()
        if 'local_z' in ele:
            rotated_ele['local_z'] = (R @ np.array(ele['local_z'])).tolist()
        rotated_elements.append(rotated_ele)
    n_nodes = original_nodes.shape[0]
    u_original = np.random.rand(6 * n_nodes)
    u_rotated = np.zeros_like(u_original)
    T = np.kron(np.eye(n_nodes), np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]]))
    u_rotated = T @ u_original
    K_original = fcn(original_nodes, elements, u_original)
    K_rotated = fcn(rotated_nodes, rotated_elements, u_rotated)
    expected_K_rotated = T @ K_original @ T.T
    assert np.allclose(K_rotated, expected_K_rotated, rtol=1e-10), 'Frame objectivity condition failed'