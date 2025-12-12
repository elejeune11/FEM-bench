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
    elements = [{'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}]
    u_zero = np.zeros(6 * n_nodes)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0.0), 'Zero displacement should produce zero stiffness matrix'
    u_nonzero = np.random.rand(6 * n_nodes)
    K = fcn(node_coords, elements, u_nonzero)
    assert np.allclose(K, K.T), 'Assembled stiffness matrix should be symmetric'
    scale = 2.5
    K_scaled = fcn(node_coords, elements, scale * u_nonzero)
    assert np.allclose(K_scaled, scale * K), 'Scaling displacements should scale K linearly'
    u1 = np.random.rand(6 * n_nodes)
    u2 = np.random.rand(6 * n_nodes)
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_sum, K1 + K2), 'Superposition should hold for displacement states'
    elements_reversed = [elements[1], elements[0]]
    K_reversed = fcn(node_coords, elements_reversed, u_nonzero)
    assert np.allclose(K, K_reversed), 'Element order should not affect the assembled matrix'

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0]])
    n_nodes = node_coords.shape[0]
    elements = [{'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}, {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-05}]
    u_original = np.random.rand(6 * n_nodes)
    K_original = fcn(node_coords, elements, u_original)
    r = R.from_euler('z', 30, degrees=True)
    R_matrix = r.as_matrix()
    node_coords_rot = (R_matrix @ node_coords.T).T
    elements_rot = []
    for el in elements:
        el_rot = el.copy()
        elements_rot.append(el_rot)
    T_block = np.block([[R_matrix, np.zeros((3, 3))], [np.zeros((3, 3)), R_matrix]])
    T_full = np.kron(np.eye(n_nodes), T_block)
    u_rot = T_full @ u_original
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    expected = T_full @ K_original @ T_full.T
    assert np.allclose(K_rot, expected, atol=1e-08), 'Frame objectivity violated under global rotation'