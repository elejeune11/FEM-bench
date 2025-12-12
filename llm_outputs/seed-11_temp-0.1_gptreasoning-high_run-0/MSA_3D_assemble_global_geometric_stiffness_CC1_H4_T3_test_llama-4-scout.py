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
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.01, 'I_z': 0.01, 'J': 0.01}, {'node_i': 1, 'node_j': 2, 'E': 1.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.01, 'I_z': 0.01, 'J': 0.01}]
    u_global_zero = np.zeros(18)
    K_g_zero = fcn(node_coords, elements, u_global_zero)
    assert np.allclose(K_g_zero, np.zeros_like(K_g_zero))
    u_global = np.random.rand(18)
    K_g = fcn(node_coords, elements, u_global)
    assert np.allclose(K_g, K_g.T)
    scale = 2.0
    K_g_scaled = fcn(node_coords, elements, scale * u_global)
    assert np.allclose(K_g_scaled, scale * K_g)
    u_global_1 = np.random.rand(18)
    u_global_2 = np.random.rand(18)
    K_g_1 = fcn(node_coords, elements, u_global_1)
    K_g_2 = fcn(node_coords, elements, u_global_2)
    K_g_sum = fcn(node_coords, elements, u_global_1 + u_global_2)
    assert np.allclose(K_g_1 + K_g_2, K_g_sum)
    elements_reversed = elements[::-1]
    K_g_reversed = fcn(node_coords, elements_reversed, u_global)
    assert np.allclose(K_g_reversed, K_g)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': 1.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.01, 'I_z': 0.01, 'J': 0.01}, {'node_i': 1, 'node_j': 2, 'E': 1.0, 'nu': 0.3, 'A': 0.1, 'I_y': 0.01, 'I_z': 0.01, 'J': 0.01}]
    u_global = np.random.rand(18)
    K_g = fcn(node_coords, elements, u_global)
    rotation_axis = np.array([1, 1, 1]) / np.sqrt(3)
    rotation_angle = np.pi / 4
    R = R.from_rotvec(rotation_angle * rotation_axis)
    node_coords_rotated = R.apply(node_coords)
    elements_rotated = []
    for element in elements:
        element_rotated = element.copy()
        element_rotated['local_z'] = R.apply(element.get('local_z', [0, 0, 1]))
        elements_rotated.append(element_rotated)
    u_global_rotated = np.concatenate([R.apply(u_global[:3]), R.apply(u_global[3:6])] * 3)
    K_g_rotated = fcn(node_coords_rotated, elements_rotated, u_global_rotated)
    T = np.eye(18)
    for i in range(0, 18, 6):
        T[i:i + 6, i:i + 6] = R.as_matrix()
        T[i + 3:i + 6, i + 3:i + 6] = R.as_matrix()
    assert np.allclose(K_g_rotated, T @ K_g @ T.T)