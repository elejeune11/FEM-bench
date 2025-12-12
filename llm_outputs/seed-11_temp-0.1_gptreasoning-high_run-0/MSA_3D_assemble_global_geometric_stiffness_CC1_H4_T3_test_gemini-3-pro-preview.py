def test_multi_element_core_correctness_assembly(fcn):
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
    props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 2e-05, 'local_z': np.array([0.0, 0.0, 1.0])}
    elements = [{**props, 'node_i': 0, 'node_j': 1}, {**props, 'node_i': 1, 'node_j': 2}]
    n_dof = 6 * len(node_coords)
    u_zero = np.zeros(n_dof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0), 'Zero displacement should result in a zero geometric stiffness matrix.'
    rng = np.random.default_rng(42)
    u_rand = rng.random(n_dof) * 0.01
    K_rand = fcn(node_coords, elements, u_rand)
    assert np.allclose(K_rand, K_rand.T), 'The geometric stiffness matrix must be symmetric.'
    scale = 2.5
    K_scaled = fcn(node_coords, elements, u_rand * scale)
    assert np.allclose(K_scaled, K_rand * scale), 'Scaling displacements should scale K_g linearly.'
    u_A = rng.random(n_dof) * 0.01
    u_B = rng.random(n_dof) * 0.01
    K_A = fcn(node_coords, elements, u_A)
    K_B = fcn(node_coords, elements, u_B)
    K_sum = fcn(node_coords, elements, u_A + u_B)
    assert np.allclose(K_sum, K_A + K_B), 'Superposition should hold for geometric stiffness assembly.'
    elements_reversed = elements[::-1]
    K_reversed = fcn(node_coords, elements_reversed, u_rand)
    assert np.allclose(K_rand, K_reversed), 'Changing element order should not affect the assembled matrix.'

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    rng = np.random.default_rng(123)
    node_coords = np.array([[0.0, 0.0, 0.0], [3.0, 4.0, 0.0]])
    local_z = np.array([0.0, 0.0, 1.0])
    elem_props = {'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0001, 'local_z': local_z}
    elements = [elem_props]
    u_global = rng.random(12) * 0.01
    K_orig = fcn(node_coords, elements, u_global)
    H = rng.random((3, 3))
    (Q, _) = np.linalg.qr(H)
    if np.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    R_mat = Q
    node_coords_rot = (R_mat @ node_coords.T).T
    local_z_rot = R_mat @ local_z
    elements_rot = [elem_props.copy()]
    elements_rot[0]['local_z'] = local_z_rot
    T = np.zeros((12, 12))
    for i in range(4):
        T[i * 3:(i + 1) * 3, i * 3:(i + 1) * 3] = R_mat
    u_global_rot = T @ u_global
    K_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    K_predicted = T @ K_orig @ T.T
    assert np.allclose(K_rot, K_predicted, atol=1e-08, rtol=1e-05), 'Geometric stiffness matrix violates frame objectivity under rigid body rotation.'