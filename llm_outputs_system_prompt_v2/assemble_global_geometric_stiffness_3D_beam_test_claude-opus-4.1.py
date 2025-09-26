def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06}]
    n_dof = 18
    u_zero = np.zeros(n_dof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0), 'Zero displacement should produce zero geometric stiffness'
    np.random.seed(42)
    u_test = np.random.randn(n_dof) * 0.01
    K_test = fcn(node_coords, elements, u_test)
    assert np.allclose(K_test, K_test.T), 'Geometric stiffness matrix must be symmetric'
    scale = 2.5
    u_scaled = scale * u_test
    K_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_scaled, scale * K_test, rtol=1e-10), 'K_g should scale linearly with displacements'
    u1 = np.random.randn(n_dof) * 0.01
    u2 = np.random.randn(n_dof) * 0.01
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2, rtol=1e-10), 'Superposition should hold for K_g'
    elements_reversed = [elements[1], elements[0]]
    K_reversed = fcn(node_coords, elements_reversed, u_test)
    assert np.allclose(K_test, K_reversed), 'Element order should not affect assembled matrix'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.5, 0.5, 0.0], [1.5, 1.0, 0.5]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06}, {'node_i': 2, 'node_j': 3, 'A': 0.01, 'I_rho': 1e-06}]
    np.random.seed(123)
    u_global = np.random.randn(24) * 0.01
    K_original = fcn(node_coords, elements, u_global)
    theta = np.pi / 6
    phi = np.pi / 4
    psi = np.pi / 3
    Rx = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    Ry = np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])
    Rz = np.array([[np.cos(psi), -np.sin(psi), 0], [np.sin(psi), np.cos(psi), 0], [0, 0, 1]])
    R = Rz @ Ry @ Rx
    node_coords_rot = node_coords @ R.T
    T = np.zeros((24, 24))
    for i in range(4):
        idx = 6 * i
        T[idx:idx + 3, idx:idx + 3] = R
        T[idx + 3:idx + 6, idx + 3:idx + 6] = R
    u_global_rot = T @ u_global
    elements_rot = []
    for ele in elements:
        ele_rot = ele.copy()
        if 'local_z' in ele:
            ele_rot['local_z'] = R @ np.array(ele['local_z'])
        elements_rot.append(ele_rot)
    K_rotated = fcn(node_coords_rot, elements_rot, u_global_rot)
    K_transformed = T @ K_original @ T.T
    assert np.allclose(K_rotated, K_transformed, rtol=1e-10, atol=1e-12), 'Frame objectivity violated: K_rot should equal T @ K @ T^T'