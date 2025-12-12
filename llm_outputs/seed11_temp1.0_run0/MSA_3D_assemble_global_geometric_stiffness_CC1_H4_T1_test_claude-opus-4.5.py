def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    element_1 = {'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0.0, 0.0, 1.0]}
    element_2 = {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0.0, 0.0, 1.0]}
    elements = [element_1, element_2]
    elements_reversed = [element_2, element_1]
    n_nodes = 3
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert K_g_zero.shape == (n_dof, n_dof), 'Output shape should be (n_dof, n_dof)'
    assert np.allclose(K_g_zero, 0.0), 'Zero displacement should produce zero stiffness matrix'
    np.random.seed(42)
    u_random = np.random.randn(n_dof) * 0.01
    K_g = fcn(node_coords, elements, u_random)
    assert np.allclose(K_g, K_g.T, atol=1e-12), 'Geometric stiffness matrix should be symmetric'
    scale_factor = 2.5
    u_scaled = scale_factor * u_random
    K_g_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_g_scaled, scale_factor * K_g, atol=1e-12), 'Scaling displacements should scale K_g linearly'
    u_state_1 = np.zeros(n_dof)
    u_state_1[0:6] = np.random.randn(6) * 0.01
    u_state_2 = np.zeros(n_dof)
    u_state_2[12:18] = np.random.randn(6) * 0.01
    K_g_1 = fcn(node_coords, elements, u_state_1)
    K_g_2 = fcn(node_coords, elements, u_state_2)
    K_g_combined = fcn(node_coords, elements, u_state_1 + u_state_2)
    assert np.allclose(K_g_combined, K_g_1 + K_g_2, atol=1e-12), 'Superposition should hold for independent displacement states'
    K_g_original_order = fcn(node_coords, elements, u_random)
    K_g_reversed_order = fcn(node_coords, elements_reversed, u_random)
    assert np.allclose(K_g_original_order, K_g_reversed_order, atol=1e-12), 'Element order should not affect the assembled result'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """

    def rotation_matrix_z(theta):
        """Rotation matrix about z-axis."""
        (c, s) = (np.cos(theta), np.sin(theta))
        return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])

    def build_global_transformation(R, n_nodes):
        """Build block-diagonal transformation matrix T."""
        n_dof = 6 * n_nodes
        T = np.zeros((n_dof, n_dof))
        for i in range(n_nodes):
            idx = 6 * i
            T[idx:idx + 3, idx:idx + 3] = R
            T[idx + 3:idx + 6, idx + 3:idx + 6] = R
        return T

    def rotate_vector(v, R):
        """Rotate a 3D vector."""
        return R @ v
    node_coords_orig = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    local_z_orig = np.array([0.0, 0.0, 1.0])
    element_1_orig = {'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06, 'local_z': local_z_orig.tolist()}
    element_2_orig = {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06, 'local_z': local_z_orig.tolist()}
    elements_orig = [element_1_orig, element_2_orig]
    n_nodes = 3
    n_dof = 6 * n_nodes
    np.random.seed(123)
    u_orig = np.random.randn(n_dof) * 0.01
    K_g_orig = fcn(node_coords_orig, elements_orig, u_orig)
    theta = np.pi / 4
    R = rotation_matrix_z(theta)
    node_coords_rot = np.array([rotate_vector(coord, R) for coord in node_coords_orig])
    local_z_rot = rotate_vector(local_z_orig, R)
    element_1_rot = {'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06, 'local_z': local_z_rot.tolist()}
    element_2_rot = {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06, 'local_z': local_z_rot.tolist()}
    elements_rot = [element_1_rot, element_2_rot]
    T = build_global_transformation(R, n_nodes)
    u_rot = T @ u_orig
    K_g_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_g_transformed = T @ K_g_orig @ T.T
    assert np.allclose(K_g_rot, K_g_transformed, atol=1e-10), 'Geometric stiffness matrix should satisfy frame objectivity: K_g^rot = T K_g T^T'