def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    n_nodes = len(node_coords)
    element_props = {'A': 0.01, 'I_rho': 1e-05, 'local_z': [0, 0, 1]}
    elements = [{'node_i': 0, 'node_j': 1, **element_props}, {'node_i': 1, 'node_j': 2, **element_props}]
    u_zero = np.zeros(6 * n_nodes)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0), 'Zero displacement should produce zero matrix'
    u_test = np.random.rand(6 * n_nodes) * 0.01
    K = fcn(node_coords, elements, u_test)
    assert np.allclose(K, K.T), 'Geometric stiffness matrix should be symmetric'
    alpha = 2.5
    K_scaled = fcn(node_coords, elements, alpha * u_test)
    assert np.allclose(K_scaled, alpha * K), 'Matrix should scale linearly with displacement'
    u1 = np.random.rand(6 * n_nodes) * 0.01
    u2 = np.random.rand(6 * n_nodes) * 0.01
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2), 'Superposition should hold for independent states'
    elements_reversed = list(reversed(elements))
    K_rev = fcn(node_coords, elements_reversed, u_test)
    assert np.allclose(K, K_rev), 'Element order should not affect result'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 1e-05, 'local_z': [0, 0, 1]}]
    u_global = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03, -0.1, 0.1, -0.2, -0.01, 0.02, -0.03])
    K_orig = fcn(node_coords, elements, u_global)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    T_block = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    T = np.kron(np.eye(2), T_block)
    node_coords_rot = node_coords @ R.T
    u_rot = T @ u_global
    K_rot = fcn(node_coords_rot, elements, u_rot)
    K_expected = T @ K_orig @ T.T
    assert np.allclose(K_rot, K_expected, rtol=1e-10), 'Frame objectivity condition failed: K_rot != T K_orig T^T'