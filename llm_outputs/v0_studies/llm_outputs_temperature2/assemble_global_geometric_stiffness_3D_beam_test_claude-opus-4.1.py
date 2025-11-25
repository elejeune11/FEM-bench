def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 0.0001}, {'node_i': 1, 'node_j': 2, 'A': 0.01, 'I_rho': 0.0001}]

    def mock_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        return np.eye(12)

    def mock_compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        return np.random.randn(12) * np.linalg.norm(u_e_global)

    def mock_local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        K = np.random.randn(12, 12)
        K = (K + K.T) / 2
        return K * (abs(Fx2) + abs(Mx2) + abs(My1) + abs(Mz1) + abs(My2) + abs(Mz2))
    import sys
    sys.modules['__main__'].beam_transformation_matrix_3D = mock_beam_transformation_matrix_3D
    sys.modules['__main__'].compute_local_element_loads_beam_3D = mock_compute_local_element_loads_beam_3D
    sys.modules['__main__'].local_geometric_stiffness_matrix_3D_beam = mock_local_geometric_stiffness_matrix_3D_beam
    u_zero = np.zeros(18)
    K_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_zero, 0), 'Zero displacement should produce zero geometric stiffness'
    u_random = np.random.randn(18) * 0.01
    K_random = fcn(node_coords, elements, u_random)
    assert np.allclose(K_random, K_random.T), 'Geometric stiffness matrix should be symmetric'
    scale = 2.5
    u_scaled = u_random * scale
    K_scaled = fcn(node_coords, elements, u_scaled)
    K_base = fcn(node_coords, elements, u_random)
    assert np.allclose(K_scaled, K_base * scale, rtol=1e-10), 'K_g should scale linearly with displacements'
    u1 = np.random.randn(18) * 0.01
    u2 = np.random.randn(18) * 0.01
    K1 = fcn(node_coords, elements, u1)
    K2 = fcn(node_coords, elements, u2)
    K_sum = fcn(node_coords, elements, u1 + u2)
    K_individual_sum = fcn(node_coords, elements, u1) + fcn(node_coords, elements, u2)
    assert np.allclose(K_sum, K_individual_sum, rtol=1e-10), 'Superposition should hold for K_g'
    elements_reversed = [elements[1], elements[0]]
    K_original = fcn(node_coords, elements, u_random)
    K_reversed = fcn(node_coords, elements_reversed, u_random)
    assert np.allclose(K_original, K_reversed), 'Element order should not affect assembled matrix'

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 0.01, 'I_rho': 0.0001}]
    theta = np.pi / 4
    R = np.array([[np.cos(theta), -np.sin(theta), 0], [np.sin(theta), np.cos(theta), 0], [0, 0, 1]])

    def mock_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        Gamma = np.eye(12)
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        if L > 1e-10:
            e_x = np.array([xj - xi, yj - yi, zj - zi]) / L
            Gamma[0:3, 0] = e_x
            Gamma[3:6, 3] = e_x
            Gamma[6:9, 6] = e_x
            Gamma[9:12, 9] = e_x
        return Gamma

    def mock_compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        F = np.zeros(12)
        F[0] = 100.0 * u_e_global[0]
        F[6] = -F[0]
        return F

    def mock_local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        K = np.zeros((12, 12))
        if abs(Fx2) > 1e-10:
            K[1, 1] = K[2, 2] = abs(Fx2) / L
            K[7, 7] = K[8, 8] = abs(Fx2) / L
            K[1, 7] = K[7, 1] = -abs(Fx2) / L
            K[2, 8] = K[8, 2] = -abs(Fx2) / L
        return K
    import sys
    sys.modules['__main__'].beam_transformation_matrix_3D = mock_beam_transformation_matrix_3D
    sys.modules['__main__'].compute_local_element_loads_beam_3D = mock_compute_local_element_loads_beam_3D
    sys.modules['__main__'].local_geometric_stiffness_matrix_3D_beam = mock_local_geometric_stiffness_matrix_3D_beam
    u_original = np.array([0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.0, 0.0])
    K_original = fcn(node_coords, elements, u_original)
    node_coords_rot = np.array([R @ node for node in node_coords])
    n_nodes = 2
    T = np.zeros((6 * n_nodes, 6 * n_nodes))
    for i in range(n_nodes):
        T[6 * i:6 * i + 3, 6 * i:6 * i + 3] = R
        T[6 * i + 3:6 * i + 6, 6 * i + 3:6 * i + 6] = R
    u_rotated = T @ u_original
    K_rotated = fcn(node_coords_rot, elements, u_rotated)
    K_expected = T @ K_original @ T.T
    assert np.allclose(K_rotated, K_expected, rtol=1e-08, atol=1e-10), 'Frame objectivity not satisfied: K_g^rot should equal T K_g T^T'