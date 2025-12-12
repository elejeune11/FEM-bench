def test_multi_element_core_correctness_assembly(fcn):
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0, 1, 0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0, 1, 0]}]
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0), 'Zero displacement should yield zero geometric stiffness matrix'
    u_nonzero = np.random.rand(n_dof)
    K = fcn(node_coords, elements, u_nonzero)
    assert np.allclose(K, K.T), 'Geometric stiffness matrix must be symmetric'
    scale = 3.5
    K_scaled = fcn(node_coords, elements, scale * u_nonzero)
    assert np.allclose(K_scaled, scale * K), 'Geometric stiffness must scale linearly with displacement'
    u1 = np.zeros(n_dof)
    u1[0] = 1.0
    u2 = np.zeros(n_dof)
    u2[6] = 1.0
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2), 'Superposition must hold for geometric stiffness'
    elements_reversed = [{'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0, 1, 0]}, {'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0, 1, 0]}]
    K_original = fcn(node_coords, elements, u_nonzero)
    K_reordered = fcn(node_coords, elements_reversed, u_nonzero)
    assert np.allclose(K_original, K_reordered), 'Element order must not affect assembled matrix'

def test_frame_objectivity_under_global_rotation(fcn):
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0, 1, 0]}]
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    R_global = R.from_euler('xyz', [0.3, 0.5, 0.2], degrees=False).as_matrix()
    node_coords_rot = node_coords @ R_global.T
    elements_rot = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': R_global @ np.array([0, 1, 0])}]
    u_global = np.random.rand(n_dof)
    u_global_rot = np.zeros(n_dof)
    for i in range(n_nodes):
        u_global_rot[6 * i:6 * i + 3] = R_global @ u_global[6 * i:6 * i + 3]
        u_global_rot[6 * i + 3:6 * i + 6] = R_global @ u_global[6 * i + 3:6 * i + 6]
    K = fcn(node_coords, elements, u_global)
    K_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R_global
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R_global
    K_transformed = T @ K @ T.T
    assert np.allclose(K_rot, K_transformed, atol=1e-10), 'Geometric stiffness must be frame objective under global rotation'