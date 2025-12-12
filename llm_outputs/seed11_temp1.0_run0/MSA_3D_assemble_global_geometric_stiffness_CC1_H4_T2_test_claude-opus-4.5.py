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
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 0.0, 1.0])}
    elements = [{**elem_props, 'nodes': (0, 1)}, {**elem_props, 'nodes': (1, 2)}]
    n_nodes = 3
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (n_dof, n_dof), 'Output shape incorrect'
    assert np.allclose(K_zero, 0.0, atol=1e-14), 'Zero displacement should produce zero K_g'
    np.random.seed(42)
    u_random = np.random.randn(n_dof) * 0.001
    K_random = fcn(node_coords, elements, u_random)
    assert np.allclose(K_random, K_random.T, atol=1e-10), 'K_g should be symmetric'
    alpha = 2.5
    u_scaled = alpha * u_random
    K_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_scaled, alpha * K_random, rtol=1e-10), 'K_g should scale linearly with displacement'
    u1 = np.random.randn(n_dof) * 0.001
    u2 = np.random.randn(n_dof) * 0.001
    K_u1 = fcn(node_coords, elements, u1)
    K_u2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K_u1 + K_u2, rtol=1e-10), 'Superposition should hold'
    elements_reversed = [{**elem_props, 'nodes': (1, 2)}, {**elem_props, 'nodes': (0, 1)}]
    K_reversed = fcn(node_coords, elements_reversed, u_random)
    assert np.allclose(K_random, K_reversed, atol=1e-14), 'Element order should not affect result'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 0.0, 1.0])}
    elements = [{**elem_props, 'nodes': (0, 1)}]
    n_nodes = 2
    n_dof = 6 * n_nodes
    np.random.seed(123)
    u_original = np.random.randn(n_dof) * 0.001
    K_original = fcn(node_coords, elements, u_original)
    angle = np.pi / 4
    rot = Rotation.from_euler('z', angle)
    R = rot.as_matrix()
    node_coords_rot = (R @ node_coords.T).T
    local_z_rot = R @ elem_props['local_z']
    elements_rot = [{**elem_props, 'nodes': (0, 1), 'local_z': local_z_rot}]
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        idx = 6 * i
        T[idx:idx + 3, idx:idx + 3] = R
        T[idx + 3:idx + 6, idx + 3:idx + 6] = R
    u_rotated = T @ u_original
    K_rotated = fcn(node_coords_rot, elements_rot, u_rotated)
    K_transformed = T @ K_original @ T.T
    assert np.allclose(K_rotated, K_transformed, rtol=1e-08, atol=1e-12), 'Frame objectivity violated: K_g^rot should equal T @ K_g @ T^T'