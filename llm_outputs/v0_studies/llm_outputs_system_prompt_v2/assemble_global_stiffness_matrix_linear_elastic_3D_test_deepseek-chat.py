def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""

    def local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J):
        k = np.eye(12)
        EA_L = E * A / L
        GJ_L = E / (2 * (1 + nu)) * J / L
        EIy_L3 = E * I_y / L ** 3
        EIz_L3 = E * I_z / L ** 3
        k[0, 0] = EA_L
        k[6, 6] = EA_L
        k[0, 6] = -EA_L
        k[6, 0] = -EA_L
        k[3, 3] = GJ_L
        k[9, 9] = GJ_L
        k[3, 9] = -GJ_L
        k[9, 3] = -GJ_L
        k[1, 1] = 12 * EIz_L3
        k[7, 7] = 12 * EIz_L3
        k[1, 7] = -12 * EIz_L3
        k[7, 1] = -12 * EIz_L3
        k[5, 5] = 4 * EIz_L3 * L ** 2
        k[11, 11] = 4 * EIz_L3 * L ** 2
        k[5, 11] = 2 * EIz_L3 * L ** 2
        k[11, 5] = 2 * EIz_L3 * L ** 2
        k[2, 2] = 12 * EIy_L3
        k[8, 8] = 12 * EIy_L3
        k[2, 8] = -12 * EIy_L3
        k[8, 2] = -12 * EIy_L3
        k[4, 4] = 4 * EIy_L3 * L ** 2
        k[10, 10] = 4 * EIy_L3 * L ** 2
        k[4, 10] = 2 * EIy_L3 * L ** 2
        k[10, 4] = 2 * EIy_L3 * L ** 2
        return k

    def beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        L = np.sqrt((xj - xi) ** 2 + (yj - yi) ** 2 + (zj - zi) ** 2)
        ex = np.array([xj - xi, yj - yi, zj - zi]) / L
        if local_z is None:
            if abs(ex[2]) > 0.9:
                local_z = np.array([0.0, 1.0, 0.0])
            else:
                local_z = np.array([0.0, 0.0, 1.0])
        ez = np.array(local_z)
        ez = ez - np.dot(ez, ex) * ex
        ez = ez / np.linalg.norm(ez)
        ey = np.cross(ez, ex)
        T_small = np.vstack([ex, ey, ez]).T
        T = np.zeros((12, 12))
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = T_small
        return T
    np.random.seed(42)
    test_cases = []
    single_element_nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    single_element = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}]
    test_cases.append((single_element_nodes, single_element, 'single_element'))
    linear_chain_nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    linear_chain_elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}]
    test_cases.append((linear_chain_nodes, linear_chain_elements, 'linear_chain'))
    triangle_nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]])
    triangle_elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}]
    test_cases.append((triangle_nodes, triangle_elements, 'triangle_loop'))
    square_nodes = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    square_elements = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}, {'node_i': 2, 'node_j': 3, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}, {'node_i': 3, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 1e-05, 'I_z': 1e-05, 'J': 1e-06}]
    test_cases.append((square_nodes, square_elements, 'square_loop'))
    for (node_coords, elements, case_name) in test_cases:
        n_nodes = node_coords.shape[0]
        expected_shape = (6 * n_nodes, 6 * n_nodes)
        K = fcn(node_coords, elements)
        assert K.shape == expected_shape, f'Shape mismatch for {case_name}: expected {expected_shape}, got {K.shape}'
        symmetry_error = np.max(np.abs(K - K.T))
        assert symmetry_error < 1e-12, f'Matrix not symmetric for {case_name}: max asymmetry = {symmetry_error}'
        for (elem_idx, element) in enumerate(elements):
            (i, j) = (element['node_i'], element['node_j'])
            block_ii = K[6 * i:6 * i + 6, 6 * i:6 * i + 6]
            block_jj = K[6 * j:6 * j + 6, 6 * j:6 * j + 6]
            block_ij = K[6 * i:6 * i + 6, 6 * j:6 * j + 6]
            block_ji = K[6 * j:6 * j + 6, 6 * i:6 * i + 6]
            assert not np.allclose(block_ii, 0), f'Zero diagonal block for node {i} in {case_name}'
            assert not np.allclose(block_jj, 0), f'Zero diagonal block for node {j} in {case_name}'
            assert not np.allclose(block_ij, 0), f'Zero off-diagonal block for element {elem_idx} in {case_name}'
            assert not np.allclose(block_ji, 0), f'Zero off-diagonal block for element {elem_idx} in {case_name}'