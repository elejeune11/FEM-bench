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
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    elements = [{'node_i': 0, 'node_j': 1, **props}, {'node_i': 1, 'node_j': 2, **props}]
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert K_g_zero.shape == (n_dof, n_dof)
    assert np.allclose(K_g_zero, np.zeros((n_dof, n_dof)))
    u1 = np.zeros(n_dof)
    u1[12] = 0.01
    K_g1 = fcn(node_coords, elements, u1)
    assert np.allclose(K_g1, K_g1.T)
    scale = 2.5
    u_scaled = scale * u1
    K_g_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_g_scaled, scale * K_g1)
    u2 = np.zeros(n_dof)
    u2[7] = 0.01
    K_g2 = fcn(node_coords, elements, u2)
    K_g_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_g_sum, K_g1 + K_g2)
    elements_rev = list(reversed(elements))
    K_g_rev = fcn(node_coords, elements_rev, u1)
    assert np.allclose(K_g1, K_g_rev)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, -1.5]])
    n_nodes = node_coords.shape[0]
    n_dof = 6 * n_nodes
    local_z_vec = np.array([0.5, -0.5, 0.5])
    local_z_vec /= np.linalg.norm(local_z_vec)
    props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}
    elements = [{'node_i': 0, 'node_j': 1, **props, 'local_z': local_z_vec}]
    np.random.seed(0)
    u_global = np.random.rand(n_dof)
    K_g_orig = fcn(node_coords, elements, u_global)
    rot_axis = np.array([1.0, 1.0, 1.0]) / np.sqrt(3)
    rot_angle = np.deg2rad(45)
    R = Rotation.from_rotvec(rot_angle * rot_axis).as_matrix()
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = [{'node_i': 0, 'node_j': 1, **props, 'local_z': R @ local_z_vec}]
    u_global_rot = np.zeros_like(u_global)
    for i in range(n_nodes):
        u_global_rot[6 * i:6 * i + 3] = R @ u_global[6 * i:6 * i + 3]
        u_global_rot[6 * i + 3:6 * i + 6] = R @ u_global[6 * i + 3:6 * i + 6]
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    R_block = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    T = np.kron(np.eye(n_nodes), R_block)
    K_g_expected = T @ K_g_orig @ T.T
    assert np.allclose(K_g_rot, K_g_expected, atol=1e-09)