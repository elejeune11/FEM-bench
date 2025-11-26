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
    props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 2e-05, 'I_z': 1e-05, 'J': 5e-05, 'local_z': np.array([0.0, 0.0, 1.0])}
    elements = [{**props, 'connectivity': [0, 1]}, {**props, 'connectivity': [1, 2]}]
    n_nodes = 3
    n_dof = n_nodes * 6
    u_zero = np.zeros(n_dof)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0), 'K_g must be zero if global displacements (and thus internal forces) are zero.'
    rng = np.random.default_rng(42)
    u_1 = rng.random(n_dof) * 0.01
    K_1 = fcn(node_coords, elements, u_1)
    assert np.allclose(K_1, K_1.T), 'Global geometric stiffness matrix must be symmetric.'
    c = 2.5
    u_scaled = c * u_1
    K_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_scaled, c * K_1), 'Geometric stiffness should scale linearly with displacement magnitude.'
    u_2 = rng.random(n_dof) * 0.01
    K_2 = fcn(node_coords, elements, u_2)
    K_sum = fcn(node_coords, elements, u_1 + u_2)
    assert np.allclose(K_sum, K_1 + K_2), 'Superposition principle should hold for geometric stiffness matrices.'
    elements_reversed = elements[::-1]
    K_reversed = fcn(node_coords, elements_reversed, u_1)
    assert np.allclose(K_1, K_reversed), 'The order of elements in the input list should not affect the assembled matrix.'

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    props = {'connectivity': [0, 1], 'E': 10000000000.0, 'nu': 0.3, 'A': 0.05, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0001, 'local_z': np.array([0.0, 0.0, 1.0])}
    elements = [props]
    rng = np.random.default_rng(101)
    u_global = rng.random(12) * 0.05
    K_g = fcn(node_coords, elements, u_global)
    theta = np.pi / 2
    (c, s) = (0.0, 1.0)
    R = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    node_coords_rot = (R @ node_coords.T).T
    props_rot = props.copy()
    props_rot['local_z'] = R @ props['local_z']
    elements_rot = [props_rot]
    blk = np.zeros((6, 6))
    blk[0:3, 0:3] = R
    blk[3:6, 3:6] = R
    T = np.zeros((12, 12))
    T[0:6, 0:6] = blk
    T[6:12, 6:12] = blk
    u_global_rot = T @ u_global
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    K_g_rot_expected = T @ K_g @ T.T
    assert np.allclose(K_g_rot, K_g_rot_expected, atol=1e-08), 'Geometric stiffness matrix should transform tensorially under global frame rotation.'