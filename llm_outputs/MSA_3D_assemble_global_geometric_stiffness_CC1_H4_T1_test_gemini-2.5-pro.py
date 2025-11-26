def test_multi_element_core_correctness_assembly(fcn):
    """Verify basic correctness of assemble_global_geometric_stiffness_3D_beam
    for a simple 3-node, 2-element chain. Checks that:
      1) zero displacement produces a zero matrix,
      2) the assembled matrix is symmetric,
      3) scaling displacements scales K_g linearly,
      4) superposition holds for independent displacement states, and
      5) element order does not affect the assembled result."""
    n_nodes = 3
    n_dof = n_nodes * 6
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 0.0, 1.0]}]

    def mock_beam_transformation_matrix_3D(*args, **kwargs):
        return np.eye(12)

    def mock_compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        return 10.0 * u_e_global

    def mock_local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        base_kg = np.arange(144).reshape(12, 12)
        base_kg = (base_kg + base_kg.T) / L
        return Fx2 * base_kg
    original_globals = fcn.__globals__.copy()
    fcn.__globals__['beam_transformation_matrix_3D'] = mock_beam_transformation_matrix_3D
    fcn.__globals__['compute_local_element_loads_beam_3D'] = mock_compute_local_element_loads_beam_3D
    fcn.__globals__['local_geometric_stiffness_matrix_3D_beam'] = mock_local_geometric_stiffness_matrix_3D_beam
    u_zero = np.zeros(n_dof)
    K_g_zero = fcn(node_coords, elements, u_zero)
    assert np.allclose(K_g_zero, np.zeros((n_dof, n_dof)))
    np.random.seed(0)
    u_global_1 = np.random.rand(n_dof)
    K_g_1 = fcn(node_coords, elements, u_global_1)
    assert np.allclose(K_g_1, K_g_1.T)
    scale_factor = 2.5
    u_global_2 = scale_factor * u_global_1
    K_g_2 = fcn(node_coords, elements, u_global_2)
    assert np.allclose(K_g_2, scale_factor * K_g_1)
    u_global_a = np.random.rand(n_dof)
    u_global_b = np.random.rand(n_dof)
    K_g_a = fcn(node_coords, elements, u_global_a)
    K_g_b = fcn(node_coords, elements, u_global_b)
    K_g_sum = fcn(node_coords, elements, u_global_a + u_global_b)
    assert np.allclose(K_g_sum, K_g_a + K_g_b)
    elements_reversed = list(reversed(elements))
    K_g_reversed = fcn(node_coords, elements_reversed, u_global_1)
    assert np.allclose(K_g_1, K_g_reversed)
    fcn.__globals__.update(original_globals)

def test_frame_objectivity_under_global_rotation(fcn):
    """Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz]."""

    def real_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        vec_x = np.array([xj - xi, yj - yi, zj - zi])
        L = np.linalg.norm(vec_x)
        if L < 1e-12:
            return np.eye(12)
        x_local = vec_x / L
        if local_z is None:
            ref_vec = np.array([0.0, 1.0, 0.0])
            if np.allclose(np.abs(x_local), [0, 1, 0]):
                ref_vec = np.array([1.0, 0.0, 0.0])
        else:
            ref_vec = np.array(local_z)
        if np.linalg.norm(np.cross(x_local, ref_vec)) < 1e-09:
            ref_vec = np.array([0.0, 1.0, 0.0]) if np.allclose(np.abs(x_local), [1, 0, 0]) else np.array([1.0, 0.0, 0.0])
        z_local_dir = np.cross(x_local, ref_vec)
        z_local = z_local_dir / np.linalg.norm(z_local_dir)
        y_local = np.cross(z_local, x_local)
        R = np.array([x_local, y_local, z_local])
        Gamma = np.zeros((12, 12))
        Gamma[0:3, 0:3] = R
        Gamma[3:6, 3:6] = R
        Gamma[6:9, 6:9] = R
        Gamma[9:12, 9:12] = R
        return Gamma

    def mock_compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        Gamma = real_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, ele.get('local_z'))
        u_e_local = Gamma @ u_e_global
        k_local = np.eye(12) * 1000000.0
        return k_local @ u_e_local

    def mock_local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        P = Fx2
        G = np.zeros((12, 12))
        G[1, 1] = G[7, 7] = 6 / 5
        G[2, 2] = G[8, 8] = 6 / 5
        G[1, 5] = G[5, 1] = L / 10
        G[1, 11] = G[11, 1] = L / 10
        G[7, 5] = G[5, 7] = -L / 10
        G[7, 11] = G[11, 7] = -L / 10
        G[2, 4] = G[4, 2] = -L / 10
        G[2, 10] = G[10, 2] = -L / 10
        G[8, 4] = G[4, 8] = L / 10
        G[8, 10] = G[10, 8] = L / 10
        G[4, 4] = G[10, 10] = 2 * L ** 2 / 15
        G[5, 5] = G[11, 11] = 2 * L ** 2 / 15
        G[4, 10] = G[10, 4] = -L ** 2 / 30
        G[5, 11] = G[11, 5] = -L ** 2 / 30
        return P / L * G
    original_globals = fcn.__globals__.copy()
    fcn.__globals__['beam_transformation_matrix_3D'] = real_beam_transformation_matrix_3D
    fcn.__globals__['compute_local_element_loads_beam_3D'] = mock_compute_local_element_loads_beam_3D
    fcn.__globals__['local_geometric_stiffness_matrix_3D_beam'] = mock_local_geometric_stiffness_matrix_3D_beam
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0.0, 1.0, 0.0]}]
    n_nodes = 2
    n_dof = n_nodes * 6
    np.random.seed(1)
    u_global = np.random.rand(n_dof)
    K_g = fcn(node_coords, elements, u_global)
    theta = np.pi / 4
    (c, s) = (np.cos(theta), np.sin(theta))
    R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    node_coords_rot = (R @ node_coords.T).T
    elements_rot = [elements[0].copy()]
    elements_rot[0]['local_z'] = R @ np.array(elements[0]['local_z'])
    u_global_rot = np.zeros_like(u_global)
    for i in range(n_nodes):
        u_i = u_global[i * 6:i * 6 + 3]
        r_i = u_global[i * 6 + 3:i * 6 + 6]
        u_global_rot[i * 6:i * 6 + 3] = R @ u_i
        u_global_rot[i * 6 + 3:i * 6 + 6] = R @ r_i
    K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
    T_node = np.block([[R, np.zeros((3, 3))], [np.zeros((3, 3)), R]])
    T = block_diag(*[T_node] * n_nodes)
    K_g_transformed = T @ K_g @ T.T
    assert np.allclose(K_g_rot, K_g_transformed, atol=1e-09)
    fcn.__globals__.update(original_globals)