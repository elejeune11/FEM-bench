def test_multi_element_core_correctness_assembly(fcn: Callable[[np.ndarray, Sequence[Dict[str, Any]], np.ndarray], np.ndarray]):
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
    dof = 6 * n_nodes
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.1, 'local_z': [0.0, 0.0, 1.0]}]
    u_zero = np.zeros(dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert K_g_zero.shape == (dof, dof)
    assert np.allclose(K_g_zero, np.zeros((dof, dof)))
    u1 = np.zeros(dof)
    u1[6] = 0.1
    u1[12] = 0.2
    u2 = np.zeros(dof)
    u2[8] = 0.05
    u2[16] = -0.05
    u_combined = u1 + u2
    K_g_combined = fcn(node_coords, elements, u_combined)
    assert K_g_combined.shape == (dof, dof)
    assert np.allclose(K_g_combined, K_g_combined.T)
    scale_factor = 2.5
    u_scaled = scale_factor * u_combined
    K_g_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_g_scaled, scale_factor * K_g_combined)
    K_g1 = fcn(node_coords, elements, u1)
    K_g2 = fcn(node_coords, elements, u2)
    assert np.allclose(fcn(node_coords, elements, u1 + u2), K_g1 + K_g2)
    elements_rev = elements[::-1]
    K_g_rev = fcn(node_coords, elements_rev, u_combined)
    assert np.allclose(K_g_combined, K_g_rev)

def test_frame_objectivity_under_global_rotation(fcn: Callable[[np.ndarray, Sequence[Dict[str, Any]], np.ndarray], np.ndarray]):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 3.0, 0.0]])
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.2, 'I_rho': 0.2, 'local_z': [1.0, 0.0, 0.0]}]
    u_global = np.zeros(dof)
    u_global[6:9] = [0.1, 0.05, 0.03]
    u_global[9:12] = [0.01, 0.02, 0.03]
    u_global[12:15] = [0.1, 0.2, -0.05]
    angle = np.pi / 4
    (c, s) = (np.cos(angle), np.sin(angle))
    v = 1 / np.sqrt(3)
    R = np.array([[c + v ** 2 * (1 - c), v ** 2 * (1 - c) - v * s, v ** 2 * (1 - c) + v * s], [v ** 2 * (1 - c) + v * s, c + v ** 2 * (1 - c), v ** 2 * (1 - c) - v * s], [v ** 2 * (1 - c) - v * s, v ** 2 * (1 - c) + v * s, c + v ** 2 * (1 - c)]])
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for ele in elements:
        ele_rot = ele.copy()
        if 'local_z' in ele:
            local_z_rot = R @ np.array(ele['local_z'])
            ele_rot['local_z'] = local_z_rot.tolist()
        elements_rot.append(ele_rot)
    u_global_rot = np.zeros_like(u_global)
    for i in range(n_nodes):
        u_global_rot[6 * i:6 * i + 3] = R @ u_global[6 * i:6 * i + 3]
        u_global_rot[6 * i + 3:6 * i + 6] = R @ u_global[6 * i + 3:6 * i + 6]
    T = np.zeros((dof, dof))
    T_node = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    for i in range(n_nodes):
        T[6 * i:6 * (i + 1), 6 * i:6 * (i + 1)] = T_node
    K_g = fcn(node_coords, elements, u_global)
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    K_g_transformed = T @ K_g @ T.T
    assert np.allclose(K_g_rot, K_g_transformed, atol=1e-09)