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
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}]
    u_global_zero = np.zeros(18)
    K_zero = fcn(node_coords, elements, u_global_zero)
    assert np.allclose(K_zero, 0), 'Zero displacement should produce a zero matrix'
    assert np.allclose(K_zero, K_zero.T), 'The assembled matrix should be symmetric'
    u_global_scaled = np.ones(18)
    K_scaled = fcn(node_coords, elements, u_global_scaled)
    assert np.allclose(K_scaled, 2 * K_zero), 'Scaling displacements should scale K_g linearly'
    u_global_superposition = np.concatenate([np.ones(6), np.zeros(12)])
    K_superposition = fcn(node_coords, elements, u_global_superposition)
    assert np.allclose(K_superposition + K_superposition, K_scaled), 'Superposition should hold for independent displacement states'
    elements_reversed = [{'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}]
    K_reversed = fcn(node_coords, elements_reversed, u_global_zero)
    assert np.allclose(K_zero, K_reversed), 'Element order should not affect the assembled result'

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}]
    u_global = np.ones(18)
    K_original = fcn(node_coords, elements, u_global)
    R = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    T = np.kron(np.eye(3), np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]]))
    node_coords_rot = node_coords @ R.T
    u_global_rot = T @ u_global
    K_rot = fcn(node_coords_rot, elements, u_global_rot)
    K_transformed = T @ K_original @ T.T
    assert np.allclose(K_rot, K_transformed), 'Frame objectivity should hold under global rotation'