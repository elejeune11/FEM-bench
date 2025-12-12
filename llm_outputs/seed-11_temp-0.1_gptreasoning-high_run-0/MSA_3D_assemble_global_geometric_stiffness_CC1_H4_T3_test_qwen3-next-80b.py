def test_multi_element_core_correctness_assembly(fcn):
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 1e-06
    J = 2e-06
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0), 'Zero displacement should yield zero geometric stiffness matrix'
    u_nonzero = np.random.rand(n_dof)
    K = fcn(node_coords, elements, u_nonzero)
    assert np.allclose(K, K.T), 'Geometric stiffness matrix must be symmetric'
    scale_factor = 3.5
    u_scaled = scale_factor * u_nonzero
    K_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_scaled, scale_factor * K), 'K_g must scale linearly with displacement'
    u1 = np.random.rand(n_dof)
    u2 = np.random.rand(n_dof)
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2), 'Superposition must hold for geometric stiffness'
    elements_reversed = [{'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    K_reordered = fcn(node_coords, elements_reversed, u_nonzero)
    assert np.allclose(K, K_reordered), 'Element order should not affect assembled K_g'

def test_frame_objectivity_under_global_rotation(fcn):
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    E = 200000000000.0
    nu = 0.3
    A = 0.01
    I_y = I_z = 1e-06
    J = 2e-06
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': [0, 0, 1]}]
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    rot = R.from_euler('xyz', [0.3, 0.5, 0.7], degrees=False)
    R_global = rot.as_matrix()
    node_coords_rot = (R_global @ node_coords.T).T
    elements_rot = []
    for elem in elements:
        elem_rot = elem.copy()
        if 'local_z' in elem:
            elem_rot['local_z'] = R_global @ np.array(elem['local_z'])
        elements_rot.append(elem_rot)
    u_global = np.random.rand(n_dof)
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R_global
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R_global
    u_global_rot = T @ u_global
    K = fcn(node_coords, elements, u_global)
    K_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    K_transformed = T @ K @ T.T
    assert np.allclose(K_rot, K_transformed, atol=1e-10), 'Geometric stiffness matrix must be frame objective'