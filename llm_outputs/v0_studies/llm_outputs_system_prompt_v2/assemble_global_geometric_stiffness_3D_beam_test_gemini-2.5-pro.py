def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.1}]
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_g_zero, 0.0)
    u_test = np.zeros(n_dof)
    u_test[12] = 0.1
    K_g = fcn(node_coords, elements, u_test)
    assert np.allclose(K_g, K_g.T)
    assert not np.allclose(K_g, 0.0)
    K_g_scaled = fcn(node_coords, elements, 2.0 * u_test)
    assert np.allclose(K_g_scaled, 2.0 * K_g)
    u1 = np.zeros(n_dof)
    u1[6] = 0.05
    u2 = np.zeros(n_dof)
    u2[12] = 0.1
    K_g1 = fcn(node_coords, elements, u1)
    K_g2 = fcn(node_coords, elements, u2)
    K_g_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_g_sum, K_g1 + K_g2)
    elements_rev = list(reversed(elements))
    K_g_rev = fcn(node_coords, elements_rev, u_test)
    assert np.allclose(K_g, K_g_rev)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.1, 'local_z': [0.0, 0.0, 1.0]}]
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    u_global = np.zeros(n_dof)
    u_global[6] = 0.1
    u_global[13] = 0.15
    K_g_orig = fcn(node_coords, elements, u_global)
    theta = np.pi / 3.0
    axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    (c, s) = (np.cos(theta), np.sin(theta))
    C = 1 - c
    (x, y, z) = axis
    R = np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s], [y * x * C + z * s, c + y * y * C, y * z * C - x * s], [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for ele in elements:
        new_ele = ele.copy()
        if 'local_z' in new_ele:
            new_ele['local_z'] = R @ np.array(new_ele['local_z'])
        elements_rot.append(new_ele)
    u_reshaped = u_global.reshape((n_nodes, 6))
    u_trans = u_reshaped[:, 0:3]
    u_rots = u_reshaped[:, 3:6]
    u_trans_rot = u_trans @ R.T
    u_rots_rot = u_rots @ R.T
    u_global_rot = np.hstack((u_trans_rot, u_rots_rot)).flatten()
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    K_g_transformed = T @ K_g_orig @ T.T
    assert np.allclose(K_g_rot, K_g_transformed, atol=1e-09)