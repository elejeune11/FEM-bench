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
    L = 2.0
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.0, 0.0], [2 * L, 0.0, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 1e-06
    Iz = 1e-06
    J = 2e-06
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': np.array([0.0, 0.0, 1.0])}]
    u_zero = np.zeros(dof)
    K0 = fcn(node_coords, elements, u_zero)
    assert K0.shape == (dof, dof)
    assert np.allclose(K0, 0.0, atol=1e-12)
    u1 = np.zeros(dof)
    u1[6 * 1 + 0] = 0.001 * L
    u1[6 * 1 + 1] = 0.0003 * L
    u1[6 * 1 + 2] = -0.0002 * L
    u1[6 * 1 + 4] = 0.002
    u1[6 * 1 + 5] = 0.001
    u1[6 * 2 + 0] = 0.002 * L
    u1[6 * 2 + 1] = -0.0004 * L
    u1[6 * 2 + 2] = 0.0001 * L
    u1[6 * 2 + 3] = -0.0015
    u1[6 * 2 + 4] = -0.0005
    u1[6 * 2 + 5] = 0.0007
    K1 = fcn(node_coords, elements, u1)
    assert K1.shape == (dof, dof)
    assert np.linalg.norm(K1) > 1e-12
    assert np.allclose(K1, K1.T, rtol=1e-09, atol=1e-10)
    alpha = 3.2
    K1_scaled = fcn(node_coords, elements, alpha * u1)
    assert np.allclose(K1_scaled, alpha * K1, rtol=1e-08, atol=1e-10)
    uB = np.zeros(dof)
    uB[6 * 0 + 1] = 0.0005 * L
    uB[6 * 0 + 5] = 0.0028
    uB[6 * 1 + 1] = -0.0002 * L
    uB[6 * 2 + 1] = 0.0001 * L
    uB[6 * 2 + 3] += 0.0004
    KA = K1
    KB = fcn(node_coords, elements, uB)
    Ksum = fcn(node_coords, elements, u1 + uB)
    assert np.allclose(Ksum, KA + KB, rtol=1e-08, atol=1e-10)
    elements_reversed = list(reversed(elements))
    Ksum_rev = fcn(node_coords, elements_reversed, u1 + uB)
    assert np.allclose(Ksum_rev, Ksum, rtol=1e-12, atol=1e-12)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    L = 3.0
    node_coords = np.array([[0.0, 0.0, 0.0], [L, 0.5 * L, 0.0], [1.7 * L, 0.7 * L, 0.0]], dtype=float)
    n_nodes = node_coords.shape[0]
    dof = 6 * n_nodes
    E = 210000000000.0
    nu = 0.3
    A = 0.012
    Iy = 1.2e-06
    Iz = 8e-07
    J = 1.9e-06
    base_local_z = np.array([0.0, 0.0, 1.0])
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': base_local_z.copy()}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': base_local_z.copy()}]
    u = np.zeros(dof)
    u[0:6] = np.array([0.001 * L, -0.0004 * L, 0.0002 * L, 0.002, -0.001, 0.0015])
    u[6:12] = np.array([0.0019 * L, 0.0002 * L, -0.0001 * L, -0.0015, 0.0023, -0.0008])
    u[12:18] = np.array([-0.0007 * L, 0.0011 * L, 0.0003 * L, 0.0006, -0.0012, 0.0004])
    K = fcn(node_coords, elements, u)
    assert K.shape == (dof, dof)
    assert np.linalg.norm(K) > 1e-14
    axis = np.array([0.3, -0.5, 0.8], dtype=float)
    axis /= np.linalg.norm(axis)
    angle = 0.6
    K_axis = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]], dtype=float)
    R = np.eye(3) + np.sin(angle) * K_axis + (1 - np.cos(angle)) * (K_axis @ K_axis)
    T = np.zeros((dof, dof), dtype=float)
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = []
    for e in elements:
        local_z_rot = R @ np.array(e['local_z'], dtype=float)
        elements_rot.append({'node_i': e['node_i'], 'node_j': e['node_j'], 'E': e['E'], 'nu': e['nu'], 'A': e['A'], 'I_y': e['I_y'], 'I_z': e['I_z'], 'J': e['J'], 'local_z': local_z_rot})
    u_rot = T @ u
    K_rot = fcn(node_coords_rot, elements_rot, u_rot)
    K_expected = T @ K @ T.T
    assert np.allclose(K_rot, K_expected, rtol=5e-07, atol=1e-09)