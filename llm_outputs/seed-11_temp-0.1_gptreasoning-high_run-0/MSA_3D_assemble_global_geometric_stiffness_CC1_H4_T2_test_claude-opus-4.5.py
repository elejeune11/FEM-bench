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
    elements = [{**elem_props, 'node_i': 0, 'node_j': 1}, {**elem_props, 'node_i': 1, 'node_j': 2}]
    n_nodes = 3
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (n_dof, n_dof), 'Output shape mismatch'
    assert np.allclose(K_zero, 0.0, atol=1e-14), 'Zero displacement should produce zero K_g'
    np.random.seed(42)
    u_rand = np.random.randn(n_dof) * 0.001
    K_rand = fcn(node_coords, elements, u_rand)
    assert np.allclose(K_rand, K_rand.T, atol=1e-10), 'K_g should be symmetric'
    scale = 2.5
    u_scaled = scale * u_rand
    K_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_scaled, scale * K_rand, atol=1e-10), 'K_g should scale linearly with displacement'
    u_a = np.zeros(n_dof)
    u_a[0] = 0.001
    u_b = np.zeros(n_dof)
    u_b[6] = 0.001
    K_a = fcn(node_coords, elements, u_a)
    K_b = fcn(node_coords, elements, u_b)
    K_ab = fcn(node_coords, elements, u_a + u_b)
    assert np.allclose(K_ab, K_a + K_b, atol=1e-10), 'Superposition should hold'
    elements_reversed = [{**elem_props, 'node_i': 1, 'node_j': 2}, {**elem_props, 'node_i': 0, 'node_j': 1}]
    K_reversed = fcn(node_coords, elements_reversed, u_rand)
    assert np.allclose(K_rand, K_reversed, atol=1e-10), 'Element order should not affect result'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords_orig = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elem_props = {'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002, 'local_z': np.array([0.0, 0.0, 1.0])}
    elements_orig = [{**elem_props, 'node_i': 0, 'node_j': 1}]
    n_nodes = 2
    n_dof = 6 * n_nodes
    np.random.seed(123)
    u_orig = np.random.randn(n_dof) * 0.001
    K_orig = fcn(node_coords_orig, elements_orig, u_orig)
    angle = np.pi / 4
    R = Rotation.from_rotvec([0, 0, angle]).as_matrix()
    node_coords_rot = (R @ node_coords_orig.T).T
    local_z_rot = R @ elem_props['local_z']
    elem_props_rot = {**elem_props, 'local_z': local_z_rot}
    elements_rot = [{**elem_props_rot, 'node_i': 0, 'node_j': 1}]
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        idx = 6 * i
        T[idx:idx + 3, idx:idx + 3] = R
        T[idx + 3:idx + 6, idx + 3:idx + 6] = R
    u_rot = T @ u_orig
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_expected = T @ K_orig @ T.T
    assert np.allclose(K_rot, K_expected, atol=1e-08), 'Frame objectivity violated: K_g^rot should equal T @ K_g @ T^T'