def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    np.random.seed(0)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = _build_chain_3node_elements(local_z=[0.0, 0.0, 1.0])
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u_zero = np.zeros(dof, dtype=float)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (dof, dof)
    assert np.allclose(K_zero, np.zeros_like(K_zero), atol=1e-12, rtol=0)
    u_rand = np.random.randn(dof) * 0.001
    K1 = fcn(node_coords, elements, u_rand)
    assert K1.shape == (dof, dof)
    assert np.allclose(K1, K1.T, atol=1e-08, rtol=1e-06)
    alpha = 2.5
    K_scaled = fcn(node_coords, elements, alpha * u_rand)
    assert np.allclose(K_scaled, alpha * K1, atol=1e-08, rtol=1e-06)
    ua = 0.3 * u_rand
    ub = 0.7 * u_rand
    K_ua = fcn(node_coords, elements, ua)
    K_ub = fcn(node_coords, elements, ub)
    K_sum = fcn(node_coords, elements, ua + ub)
    assert np.allclose(K_sum, K_ua + K_ub, atol=1e-08, rtol=1e-06)
    elements_shuffled = list(reversed(elements))
    K_shuffled = fcn(node_coords, elements_shuffled, u_rand)
    assert np.allclose(K_shuffled, K1, atol=1e-08, rtol=1e-06)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    np.random.seed(1)
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements = _build_chain_3node_elements(local_z=[0.0, 0.0, 1.0])
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    u = np.random.randn(dof) * 0.001
    K = fcn(node_coords, elements, u)
    assert K.shape == (dof, dof)
    axis = np.array([1.0, 1.0, 0.5])
    angle = 0.437
    R = _rotation_matrix_axis_angle(axis, angle)
    node_coords_rot = node_coords @ R.T
    elements_rot = []
    for ele in elements:
        ele_rot = ele.copy()
        if 'local_z' in ele:
            ele_rot['local_z'] = (R @ np.asarray(ele['local_z'], dtype=float)).astype(float)
        elements_rot.append(ele_rot)
    T = np.zeros((dof, dof), dtype=float)
    for i in range(n_nodes):
        idx = 6 * i
        T[idx:idx + 3, idx:idx + 3] = R
        T[idx + 3:idx + 6, idx + 3:idx + 6] = R
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_transformed = T @ K @ T.T
    assert np.allclose(K_rot, K_transformed, atol=1e-06, rtol=1e-06)