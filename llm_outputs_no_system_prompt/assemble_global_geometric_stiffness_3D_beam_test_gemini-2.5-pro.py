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
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 0.1}]

    def mock_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        return np.eye(12)

    def mock_compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        L = np.linalg.norm(np.array([xj, yj, zj]) - np.array([xi, yi, zi]))
        axial_stretch = u_e_global[6] - u_e_global[0]
        axial_force = 10.0 * axial_stretch
        loads = np.zeros(12)
        loads[6] = axial_force
        return loads

    def mock_local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        k_g = np.zeros((12, 12))
        if not np.isclose(Fx2, 0):
            val = Fx2 / L
            k_g[1, 1] = val
            k_g[7, 7] = val
            k_g[1, 7] = -val
            k_g[7, 1] = -val
            k_g[2, 2] = val
            k_g[8, 8] = val
            k_g[2, 8] = -val
            k_g[8, 2] = -val
        return k_g
    fcn_module_name = fcn.__module__
    patch_target_1 = f'{fcn_module_name}.beam_transformation_matrix_3D'
    patch_target_2 = f'{fcn_module_name}.compute_local_element_loads_beam_3D'
    patch_target_3 = f'{fcn_module_name}.local_geometric_stiffness_matrix_3D_beam'
    with patch(patch_target_1, new=mock_beam_transformation_matrix_3D), patch(patch_target_2, new=mock_compute_local_element_loads_beam_3D), patch(patch_target_3, new=mock_local_geometric_stiffness_matrix_3D_beam):
        u_zero = np.zeros(n_dof)
        K_g_zero = fcn(node_coords, elements, u_zero)
        assert np.all(K_g_zero == 0)
        assert K_g_zero.shape == (n_dof, n_dof)
        u1 = np.zeros(n_dof)
        u1[6] = 0.1
        u2 = np.zeros(n_dof)
        u2[12] = -0.05
        K_g1 = fcn(node_coords, elements, u1)
        assert np.allclose(K_g1, K_g1.T)
        scale_factor = -2.0
        u1_scaled = scale_factor * u1
        K_g1_scaled = fcn(node_coords, elements, u1_scaled)
        assert np.allclose(K_g1_scaled, scale_factor * K_g1)
        K_g2 = fcn(node_coords, elements, u2)
        K_g_sum_separate = K_g1 + K_g2
        K_g_sum_combined = fcn(node_coords, elements, u1 + u2)
        assert np.allclose(K_g_sum_combined, K_g_sum_separate)
        elements_rev = list(reversed(elements))
        K_g_rev = fcn(node_coords, elements_rev, u1 + u2)
        assert np.allclose(K_g_sum_combined, K_g_rev)

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 2.0, 3.0]])
    n_nodes = len(node_coords)
    n_dof = 6 * n_nodes
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': np.array([0.0, 1.0, 0.0])}]
    u_global = np.random.default_rng(0).random(n_dof)
    R_mat = Rotation.from_rotvec(np.deg2rad(30) * np.array([1, 1, 1]) / np.sqrt(3)).as_matrix()
    R_block = np.block([[R_mat, np.zeros((3, 3))], [np.zeros((3, 3)), R_mat]])
    T = np.kron(np.eye(n_nodes, dtype=int), R_block)
    node_coords_rot = (R_mat @ node_coords.T).T
    elements_rot = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 0.1, 'local_z': R_mat @ elements[0]['local_z']}]
    u_global_rot = T @ u_global
    Gamma_orig = Rotation.from_euler('xyz', [10, 20, 30], degrees=True).as_matrix()
    Gamma_orig_block = np.block([[Gamma_orig, np.zeros((3, 3))], [np.zeros((3, 3)), Gamma_orig]])
    Gamma_orig_12x12 = np.kron(np.eye(2, dtype=int), Gamma_orig_block)

    def mock_beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        coords_i = np.array([xi, yi, zi])
        is_rotated = np.allclose(coords_i, node_coords_rot[0])
        if is_rotated:
            return Gamma_orig_12x12 @ T.T
        else:
            return Gamma_orig_12x12

    def mock_compute_local_element_loads_beam_3D(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        return np.arange(1, 13, dtype=float)

    def mock_local_geometric_stiffness_matrix_3D_beam(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        k_g = np.arange(144).reshape(12, 12)
        return k_g + k_g.T
    fcn_module_name = fcn.__module__
    patch_target_1 = f'{fcn_module_name}.beam_transformation_matrix_3D'
    patch_target_2 = f'{fcn_module_name}.compute_local_element_loads_beam_3D'
    patch_target_3 = f'{fcn_module_name}.local_geometric_stiffness_matrix_3D_beam'
    with patch(patch_target_1, new=mock_beam_transformation_matrix_3D), patch(patch_target_2, new=mock_compute_local_element_loads_beam_3D), patch(patch_target_3, new=mock_local_geometric_stiffness_matrix_3D_beam):
        K_g_orig = fcn(node_coords, elements, u_global)
        K_g_rot = fcn(node_coords_rot, elements_rot, u_global_rot)
        K_g_transformed = T @ K_g_orig @ T.T
        assert np.allclose(K_g_rot, K_g_transformed)