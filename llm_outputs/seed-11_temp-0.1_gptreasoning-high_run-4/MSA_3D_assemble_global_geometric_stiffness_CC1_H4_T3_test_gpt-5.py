def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness for a simple 3-node, 2-element chain.
    Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result.
    """
    L = 2.0
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 1.2e-06
    Iz = 8e-07
    J = 2.5e-06
    z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': z_ref}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': z_ref}]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u_zero = np.zeros(ndof, dtype=float)
    K_zero = fcn(node_coords, elements, u_zero)
    assert K_zero.shape == (ndof, ndof)
    assert np.allclose(K_zero, 0.0, rtol=0.0, atol=1e-14)
    u1 = np.array([0.0, 0.0005, -0.0002, 0.001, 0.0, 0.0003, 0.002, -0.0003, 0.0001, -0.0002, 0.0004, -0.0001, 0.004, 0.0004, -0.0005, 0.0006, -0.0003, 0.0002], dtype=float)
    K1 = fcn(node_coords, elements, u1)
    assert np.allclose(K1, K1.T, rtol=1e-09, atol=1e-12)
    s = 2.4
    K1_scaled = fcn(node_coords, elements, s * u1)
    assert np.allclose(K1_scaled, s * K1, rtol=1e-08, atol=1e-12)
    u2 = np.array([0.0003, -0.0002, 0.0001, -0.0004, 0.0005, -0.0006, -0.0005, 0.0006, -0.0004, 0.0003, -0.0002, 0.0001, 0.0002, -0.0001, 0.0003, -0.0002, 0.0004, -0.0003], dtype=float)
    K2 = fcn(node_coords, elements, u2)
    K12 = fcn(node_coords, elements, u1 + u2)
    assert np.allclose(K12, K1 + K2, rtol=1e-08, atol=1e-12)
    elements_reversed = list(reversed(elements))
    K1_rev = fcn(node_coords, elements_reversed, u1)
    assert np.allclose(K1_rev, K1, rtol=1e-12, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity: rotating geometry, local axes, and displacement by R yields K_g^rot ≈ T K_g T^T."""

    def rotation_matrix_from_axis_angle(axis, angle):
        axis = np.asarray(axis, dtype=float)
        axis = axis / np.linalg.norm(axis)
        x, y, z = axis
        K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=float)
        I = np.eye(3)
        return I + np.sin(angle) * K + (1.0 - np.cos(angle)) * (K @ K)
    L = 1.5
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [L, L, 0.0]], dtype=float)
    E = 210000000000.0
    nu = 0.3
    A = 0.012
    Iy = 1e-06
    Iz = 1.1e-06
    J = 2e-06
    z_ref = np.array([0.0, 0.0, 1.0], dtype=float)
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': z_ref}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': z_ref}]
    n_nodes = node_coords.shape[0]
    ndof = 6 * n_nodes
    u = np.array([0.0003, 0.0002, -0.0001, 0.001, -0.0005, 0.0004, -0.0002, 0.0003, 0.0005, -0.0006, 0.0002, -0.0015, 0.0001, -0.0004, 0.0002, 0.0003, 0.0007, -0.0002], dtype=float)
    K = fcn(node_coords, elements, u)
    assert K.shape == (ndof, ndof)
    axis = np.array([1.0, 0.5, -0.75], dtype=float)
    angle = 0.7
    R = rotation_matrix_from_axis_angle(axis, angle)
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for e in elements:
        z_rot = R @ np.asarray(e['local_z'], dtype=float)
        e_rot = dict(e)
        e_rot['local_z'] = z_rot
        elements_rot.append(e_rot)
    T = np.zeros((ndof, ndof), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_pred = T @ K @ T.T
    assert np.allclose(K_rot, K_pred, rtol=2e-08, atol=2e-11)