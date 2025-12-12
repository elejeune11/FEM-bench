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
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}]
    n_nodes = 3
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (n_dof, n_dof), 'Output shape mismatch'
    assert np.allclose(K_zero, 0.0), 'Zero displacement should produce zero geometric stiffness'
    u_nonzero = np.random.RandomState(42).randn(n_dof) * 0.01
    K_nonzero = fcn(node_coords, elements, u_nonzero)
    assert np.allclose(K_nonzero, K_nonzero.T), 'Geometric stiffness matrix should be symmetric'
    scale = 2.5
    u_scaled = scale * u_nonzero
    K_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_scaled, scale * K_nonzero, rtol=1e-10), 'K_g should scale linearly with displacement'
    u_a = np.zeros(n_dof)
    u_a[0:6] = np.array([0.001, 0.002, 0.0, 0.0, 0.0, 0.001])
    u_b = np.zeros(n_dof)
    u_b[12:18] = np.array([0.001, -0.001, 0.0, 0.0, 0.001, 0.0])
    K_a = fcn(node_coords, elements, u_a)
    K_b = fcn(node_coords, elements, u_b)
    K_ab = fcn(node_coords, elements, u_a + u_b)
    assert np.allclose(K_ab, K_a + K_b, rtol=1e-10), 'Superposition should hold for geometric stiffness'
    elements_reversed = [{'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}, {'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}]
    K_reversed = fcn(node_coords, elements_reversed, u_nonzero)
    assert np.allclose(K_nonzero, K_reversed, rtol=1e-10), 'Element order should not affect assembled result'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords_orig = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_orig = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06, 'local_z': [0, 0, 1]}]
    n_nodes = 2
    n_dof = 6 * n_nodes
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    node_coords_rot = (R @ node_coords_orig.T).T
    local_z_orig = np.array([0, 0, 1])
    local_z_rot = R @ local_z_orig
    elements_rot = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-06, 'local_z': local_z_rot.tolist()}]
    np.random.seed(123)
    u_orig = np.random.randn(n_dof) * 0.01

    def build_T(R, n_nodes):
        T = np.zeros((6 * n_nodes, 6 * n_nodes))
        for i in range(n_nodes):
            idx = 6 * i
            T[idx:idx + 3, idx:idx + 3] = R
            T[idx + 3:idx + 6, idx + 3:idx + 6] = R
        return T
    T = build_T(R, n_nodes)
    u_rot = T @ u_orig
    K_orig = fcn(node_coords_orig, elements_orig, u_orig)
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_expected = T @ K_orig @ T.T
    assert np.allclose(K_rot, K_expected, rtol=1e-08, atol=1e-12), 'Geometric stiffness should satisfy frame objectivity: K_rot = T @ K_orig @ T^T'