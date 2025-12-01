def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
for a simple 3-node, 2-element chain. Checks that:
  1) zero displacement produces a zero matrix,
  2) the assembled matrix is symmetric,
  3) scaling displacements scales K_g linearly,
  4) superposition holds for independent displacement states, and
  5) element order does not affect the assembled result."""
    n_nodes = 3
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elem_props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003, 'local_z': np.array([0.0, 0.0, 1.0])}
    elements = [{'nodes': (0, 1), **elem_props}, {'nodes': (1, 2), **elem_props}]
    n_dof = 6 * n_nodes
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert np.all(K_g_zero == 0), 'Zero displacement should yield a zero K_g'
    u1 = np.zeros(n_dof)
    u1[12] = 0.01
    K_g1 = fcn(node_coords, elements, u1)
    assert np.allclose(K_g1, K_g1.T), 'K_g must be symmetric'
    scale = 2.5
    u_scaled = scale * u1
    K_g_scaled = fcn(node_coords, elements, u_scaled)
    assert np.allclose(K_g_scaled, scale * K_g1), 'K_g should scale linearly with displacement'
    u2 = np.zeros(n_dof)
    u2[15] = 0.1
    K_g2 = fcn(node_coords, elements, u2)
    u_sum = u1 + u2
    K_g_sum = fcn(node_coords, elements, u_sum)
    assert np.allclose(K_g_sum, K_g1 + K_g2), 'Superposition should hold for K_g'
    elements_reversed = [elements[1], elements[0]]
    K_g_reversed = fcn(node_coords, elements_reversed, u1)
    assert np.allclose(K_g_reversed, K_g1), 'Element order should not affect the final assembled K_g'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
Rotating the entire system (geometry, local axes, and displacement field) by
a global rotation R should produce a geometric stiffness matrix K_g^rot that
satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    n_nodes = 2
    node_coords = np.array([[0.0, 0.0, 0.0], [2.0, 1.0, -1.0]])
    beam_axis = node_coords[1] - node_coords[0]
    local_z_unnorm = np.cross(beam_axis, np.array([0.0, 0.0, 1.0]))
    if np.linalg.norm(local_z_unnorm) < 1e-06:
        local_z_unnorm = np.cross(beam_axis, np.array([0.0, 1.0, 0.0]))
    local_z = local_z_unnorm / np.linalg.norm(local_z_unnorm)
    elem_props = {'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0002, 'J': 0.0003, 'local_z': local_z}
    elements = [{'nodes': (0, 1), **elem_props}]
    n_dof = 6 * n_nodes
    np.random.seed(42)
    u_global = np.random.rand(n_dof) * 0.05
    K_g = fcn(node_coords, elements, u_global)
    rot_axis = np.random.rand(3)
    rot_axis /= np.linalg.norm(rot_axis)
    R_obj = Rotation.from_rotvec(np.deg2rad(30) * rot_axis)
    R = R_obj.as_matrix()
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = copy.deepcopy(elements)
    elements_rot[0]['local_z'] = R @ elements[0]['local_z']
    u_global_reshaped = u_global.reshape(n_nodes, 6)
    u_trans = u_global_reshaped[:, :3]
    u_rots = u_global_reshaped[:, 3:]
    u_trans_rot = (R @ u_trans.T).T
    u_rots_rot = (R @ u_rots.T).T
    u_global_rot = np.hstack((u_trans_rot, u_rots_rot)).flatten()
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    T_node = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    T = block_diag(*[T_node] * n_nodes)
    K_g_expected = T @ K_g @ T.T
    assert np.allclose(K_g_rot, K_g_expected, atol=1e-09), 'K_g does not satisfy frame objectivity under global rotation'