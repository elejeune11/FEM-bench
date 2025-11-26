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

    def mock_beam_transformation(xi, yi, zi, xj, yj, zj, local_z=None):
        return np.eye(12)

    def mock_compute_loads(ele, xi, yi, zi, xj, yj, zj, u_e_global):
        return u_e_global.astype(float)

    def mock_local_geometric_stiffness(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        mat = np.zeros((12, 12))
        idx = np.arange(12)
        mat[idx, idx] = 1.0
        mat[0, 6] = mat[6, 0] = -0.5
        factor = Fx2 + Mx2 + My1 + Mz1 + My2 + Mz2
        return mat * factor
    nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0}]
    n_dof = 3 * 6
    u_zero = np.zeros(n_dof)
    u_rand1 = np.random.rand(n_dof)
    u_rand2 = np.random.rand(n_dof)
    mod_name = fcn.__module__
    with patch(f'{mod_name}.beam_transformation_matrix_3D', side_effect=mock_beam_transformation), patch(f'{mod_name}.compute_local_element_loads_beam_3D', side_effect=mock_compute_loads), patch(f'{mod_name}.local_geometric_stiffness_matrix_3D_beam', side_effect=mock_local_geometric_stiffness):
        K_0 = fcn(nodes, elements, u_zero)
        assert np.allclose(K_0, 0), 'Geometric stiffness should be zero for zero displacement.'
        K_1 = fcn(nodes, elements, u_rand1)
        assert np.allclose(K_1, K_1.T), 'Global geometric stiffness matrix must be symmetric.'
        alpha = 2.5
        K_scaled = fcn(nodes, elements, u_rand1 * alpha)
        assert np.allclose(K_scaled, K_1 * alpha), 'Matrix should scale linearly with displacements (given linear elastic assumption).'
        K_2 = fcn(nodes, elements, u_rand2)
        K_sum = fcn(nodes, elements, u_rand1 + u_rand2)
        assert np.allclose(K_sum, K_1 + K_2), 'Superposition principle should hold.'
        K_rev = fcn(nodes, elements[::-1], u_rand1)
        assert np.allclose(K_1, K_rev), 'Order of elements should not affect assembly result.'

def test_frame_objectivity_under_global_rotation(fcn):
    """
    Verify frame objectivity of assemble_global_geometric_stiffness_3D_beam.
    Rotating the entire system (geometry, local axes, and displacement field) by
    a global rotation R should produce a geometric stiffness matrix K_g^rot that
    satisfies: K_g^rot ≈ T K_g T^T, where T is block-diagonal with per-node blocks
    diag(R, R) acting on [u_x,u_y,u_z, rx,ry,rz].
    """

    def mock_beam_transformation(xi, yi, zi, xj, yj, zj, local_z=None):
        vec = np.array([xj - xi, yj - yi, zj - zi])
        L = np.linalg.norm(vec)
        if L < 1e-09:
            return np.eye(12)
        v1 = vec / L
        if local_z is not None:
            lz = np.array(local_z)
            v2 = lz - np.dot(lz, v1) * v1
            if np.linalg.norm(v2) < 1e-09:
                v2 = np.array([0, 1, 0]) if abs(v1[2]) < 0.9 else np.array([1, 0, 0])
        else:
            v2 = np.array([0, 1, 0]) if abs(v1[2]) < 0.9 else np.array([1, 0, 0])
        v2 /= np.linalg.norm(v2)
        v3 = np.cross(v1, v2)
        R = np.column_stack((v1, v2, v3))
        Gamma = np.zeros((12, 12))
        for k in range(4):
            Gamma[3 * k:3 * k + 3, 3 * k:3 * k + 3] = R
        return Gamma

    def mock_compute_loads(ele, xi, yi, zi, xj, yj, zj, u_g):
        Gamma = mock_beam_transformation(xi, yi, zi, xj, yj, zj, ele.get('local_z'))
        u_l = Gamma.T @ u_g
        return u_l * 100.0

    def mock_local_geometric_stiffness(L, A, I_rho, Fx2, Mx2, My1, Mz1, My2, Mz2):
        base = np.eye(12)
        base[0, 1] = base[1, 0] = 0.5
        s = Fx2 + Mx2
        return base * s
    nodes = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [2.0, 1.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0, 1, 0]}, {'node_i': 1, 'node_j': 2, 'A': 1.0, 'I_rho': 1.0, 'local_z': [0, 0, 1]}]
    u_global = np.random.rand(len(nodes) * 6)
    mod_name = fcn.__module__
    with patch(f'{mod_name}.beam_transformation_matrix_3D', side_effect=mock_beam_transformation), patch(f'{mod_name}.compute_local_element_loads_beam_3D', side_effect=mock_compute_loads), patch(f'{mod_name}.local_geometric_stiffness_matrix_3D_beam', side_effect=mock_local_geometric_stiffness):
        K_ref = fcn(nodes, elements, u_global)
        R_rot = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
        nodes_rot = (R_rot @ nodes.T).T
        elements_rot = []
        for el in elements:
            new_el = el.copy()
            if 'local_z' in new_el:
                new_el['local_z'] = R_rot @ np.array(new_el['local_z'])
            elements_rot.append(new_el)
        n_nodes = len(nodes)
        u_rot = np.zeros_like(u_global)
        T_sys = np.zeros((n_nodes * 6, n_nodes * 6))
        for i in range(n_nodes):
            idx = i * 6
            u_rot[idx:idx + 3] = R_rot @ u_global[idx:idx + 3]
            u_rot[idx + 3:idx + 6] = R_rot @ u_global[idx + 3:idx + 6]
            T_sys[idx:idx + 3, idx:idx + 3] = R_rot
            T_sys[idx + 3:idx + 6, idx + 3:idx + 6] = R_rot
        K_rot = fcn(nodes_rot, elements_rot, u_rot)
        K_expected = T_sys @ K_ref @ T_sys.T
        assert np.allclose(K_rot, K_expected, atol=1e-08), 'Assembly failed frame objectivity test under 90-degree rotation.'