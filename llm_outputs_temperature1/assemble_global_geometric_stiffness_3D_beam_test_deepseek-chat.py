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
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 0.0001, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 0.0001, 'local_z': [0, 0, 1]}]
    n_nodes = 3
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_g_zero, 0), 'Zero displacement should produce zero geometric stiffness'
    u_test = np.random.rand(n_dof) * 0.01
    K_g = fcn(node_coords, elements, u_test)
    assert np.allclose(K_g, K_g.T), 'Geometric stiffness matrix should be symmetric'
    scale = 2.0
    K_g_scaled = fcn(node_coords, elements, scale * u_test)
    assert np.allclose(K_g_scaled, scale * K_g), 'K_g should scale linearly with displacement'
    u1 = np.random.rand(n_dof) * 0.01
    u2 = np.random.rand(n_dof) * 0.01
    K_g_u1 = fcn(node_coords, elements, u1)
    K_g_u2 = fcn(node_coords, elements, u2)
    K_g_u1_plus_u2 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K_g_u1_plus_u2, K_g_u1 + K_g_u2), 'Superposition should hold for K_g'
    elements_reversed = list(reversed(elements))
    K_g_reversed = fcn(node_coords, elements_reversed, u_test)
    assert np.allclose(K_g_reversed, K_g), 'Result should be independent of element order'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords_orig = np.array([[0, 0, 0], [1, 0, 0], [1, 1, 0]])
    elements_orig = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 0.0001, 'local_z': [0, 0, 1]}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 0.0001, 'local_z': [0, 0, 1]}]
    n_nodes = 3
    n_dof = 6 * n_nodes
    u_global_orig = np.random.rand(n_dof) * 0.01
    K_g_orig = fcn(node_coords_orig, elements_orig, u_global_orig)
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])
    node_coords_rot = node_coords_orig @ R.T
    elements_rot = []
    for ele in elements_orig:
        ele_rot = ele.copy()
        if 'local_z' in ele:
            local_z_rot = R @ np.array(ele['local_z'])
            ele_rot['local_z'] = local_z_rot.tolist()
        elements_rot.append(ele_rot)
    u_global_rot = np.zeros_like(u_global_orig)
    for i in range(n_nodes):
        trans_dofs = slice(6 * i, 6 * i + 3)
        u_global_rot[trans_dofs] = R @ u_global_orig[trans_dofs]
        rot_dofs = slice(6 * i + 3, 6 * i + 6)
        u_global_rot[rot_dofs] = R @ u_global_orig[rot_dofs]
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    T = np.zeros((n_dof, n_dof))
    for i in range(n_nodes):
        block_start = 6 * i
        block_end = 6 * i + 6
        T[block_start:block_end, block_start:block_end] = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    K_g_transformed = T @ K_g_orig @ T.T
    assert np.allclose(K_g_rot, K_g_transformed, atol=1e-10), 'Frame objectivity violation: K_g should transform correctly under rotation'