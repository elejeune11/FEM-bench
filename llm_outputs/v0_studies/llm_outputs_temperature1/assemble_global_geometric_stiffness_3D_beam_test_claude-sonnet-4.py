def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    import numpy as np
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': None}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.1, 'local_z': None}]
    u_zero = np.zeros(18)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0), 'Zero displacement should produce zero matrix'
    u_test = np.random.rand(18) * 0.1
    K_test = fcn(node_coords, elements, u_test)
    assert np.allclose(K_test, K_test.T), 'Assembled matrix should be symmetric'
    scale = 2.5
    u_scaled = scale * u_test
    K_scaled = fcn(node_coords, elements, u_scaled)
    K_expected = scale * K_test
    assert np.allclose(K_scaled, K_expected), 'Scaling displacement should scale K_g linearly'
    u_test2 = np.random.rand(18) * 0.1
    K_test2 = fcn(node_coords, elements, u_test2)
    K_combined = fcn(node_coords, elements, u_test + u_test2)
    K_sum = K_test + K_test2
    assert np.allclose(K_combined, K_sum), 'Superposition should hold'
    elements_reordered = [elements[1], elements[0]]
    K_reordered = fcn(node_coords, elements_reordered, u_test)
    assert np.allclose(K_test, K_reordered), 'Element order should not affect result'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    import numpy as np
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.5, 0.0], [2.0, 0.0, 1.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.1, 'local_z': [0, 0, 1]}]
    u_original = np.random.rand(18) * 0.1
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for ele in elements:
        ele_rot = ele.copy()
        if ele['local_z'] is not None:
            ele_rot['local_z'] = R @ np.array(ele['local_z'])
        elements_rot.append(ele_rot)
    n_nodes = len(node_coords)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        idx = 6 * i
        T[idx:idx + 3, idx:idx + 3] = R
        T[idx + 3:idx + 6, idx + 3:idx + 6] = R
    u_rotated = T @ u_original
    K_original = fcn(node_coords, elements, u_original)
    K_rotated = fcn(node_coords_rot, elements_rot, u_rotated)
    K_expected = T @ K_original @ T.T
    assert np.allclose(K_rotated, K_expected, atol=1e-10), 'Frame objectivity should hold under global rotation'