def test_multi_element_core_correctness_assembly(fcn):
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    n_nodes = len(node_coords)
    elements = [{'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 1, 0]}, {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 1, 0]}]
    u_zero = np.zeros(6 * n_nodes)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0), 'Zero displacement should produce zero geometric stiffness matrix'
    u_nonzero = np.random.rand(6 * n_nodes)
    K = fcn(node_coords, elements, u_nonzero)
    assert np.allclose(K, K.T), 'Geometric stiffness matrix must be symmetric'
    scale_factor = 3.5
    u_scaled = scale_factor * u_nonzero
    K_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_scaled, scale_factor * K), 'Scaling displacement should scale K_g linearly'
    u1 = np.random.rand(6 * n_nodes)
    u2 = np.random.rand(6 * n_nodes)
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2), 'Superposition must hold for geometric stiffness matrix'
    elements_reversed = [elements[1], elements[0]]
    K_reordered = fcn(node_coords, elements_reversed, u_nonzero)
    assert np.allclose(K, K_reordered), 'Element order should not affect assembled matrix'

def test_frame_objectivity_under_global_rotation(fcn):
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    n_nodes = len(node_coords)
    elements = [{'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-06, 'I_z': 1e-06, 'J': 2e-06, 'local_z': [0, 1, 0]}]
    u_global = np.random.rand(6 * n_nodes)
    K_original = fcn(node_coords, elements, u_global)
    R_global = R.from_euler('zyx', np.random.rand(3) * 2 * np.pi).as_matrix()
    node_coords_rot = node_coords @ R_global.T
    local_z_rot = R_global @ np.array(elements[0]['local_z'])
    T_block = np.block([[R_global, np.zeros((3, 3))], [np.zeros((3, 3)), R_global]])
    T_full = np.kron(np.eye(n_nodes), T_block)
    u_global_rot = T_full @ u_global
    elements_rot = [{'E': elements[0]['E'], 'nu': elements[0]['nu'], 'A': elements[0]['A'], 'I_y': elements[0]['I_y'], 'I_z': elements[0]['I_z'], 'J': elements[0]['J'], 'local_z': local_z_rot.tolist()}]
    K_rotated = fcn(node_coords_rot, elements_rot, u_global_rot)
    K_transformed = T_full @ K_original @ T_full.T
    assert np.allclose(K_rotated, K_transformed, atol=1e-10), 'Geometric stiffness matrix must be frame objective'