def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 0.0001}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 0.0001}]
    n_dof = 18
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_g_zero, 0.0), 'Zero displacement should produce zero geometric stiffness'
    np.random.seed(42)
    u_random = np.random.randn(n_dof) * 0.01
    K_g = fcn(node_coords, elements, u_random)
    assert np.allclose(K_g, K_g.T), 'Geometric stiffness matrix must be symmetric'
    scale_factor = 2.5
    u_scaled = u_random * scale_factor
    K_g_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_g_scaled, K_g * scale_factor), 'K_g should scale linearly with displacements'
    u1 = np.random.randn(n_dof) * 0.01
    u2 = np.random.randn(n_dof) * 0.01
    K_g1 = fcn(node_coords, elements, u1)
    K_g2 = fcn(node_coords, elements, u2)
    K_g_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_g_sum, K_g1 + K_g2), 'Superposition should hold for K_g'
    elements_reversed = [elements[1], elements[0]]
    K_g_reversed = fcn(node_coords, elements_reversed, u_random)
    assert np.allclose(K_g, K_g_reversed), 'Element order should not affect assembled matrix'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.5, 0.0], [1.5, 1.0, 0.5]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 0.0001}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 0.0001}, {'node_i': 2, 'node_j': 3, 'A': 0.01, 'I_rho': 0.0001}]
    np.random.seed(123)
    u_global = np.random.randn(24) * 0.01
    K_g_original = fcn(node_coords, elements, u_global)
    theta = np.pi / 6
    phi = np.pi / 4
    psi = np.pi / 3
    Rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    Ry = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for ele in elements:
        ele_rot = ele.copy()
        if 'local_z' in ele:
            ele_rot['local_z'] = R @ np.array(ele['local_z'])
        elements_rot.append(ele_rot)
    n_nodes = len(node_coords)
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    u_global_rot = T @ u_global
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    K_g_expected = T @ K_g_original @ T.T
    assert np.allclose(K_g_rot, K_g_expected, rtol=1e-10, atol=1e-12), 'Rotated K_g should equal T @ K_g @ T^T for frame objectivity'