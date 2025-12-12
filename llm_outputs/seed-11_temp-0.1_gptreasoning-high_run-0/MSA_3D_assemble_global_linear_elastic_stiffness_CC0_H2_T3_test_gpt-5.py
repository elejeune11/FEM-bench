def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 8.333e-06
    Iz = 8.333e-06
    J = 1.667e-05

    def build_elements(edges):
        return [{'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J} for (i, j) in edges]

    def check_basic_properties(K, node_coords):
        n_nodes = node_coords.shape[0]
        assert K.shape == (6 * n_nodes, 6 * n_nodes)
        assert np.allclose(K, K.T, atol=1e-09, rtol=1e-09)

    def idx_block(a):
        return slice(6 * a, 6 * a + 6)

    def check_element_contributions(K, elements):
        for el in elements:
            i = el['node_i']
            j = el['node_j']
            block_ij = K[idx_block(i), idx_block(j)]
            block_ji = K[idx_block(j), idx_block(i)]
            assert np.max(np.abs(block_ij)) > 0.0
            assert np.max(np.abs(block_ji)) > 0.0
            idx = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
            block_12 = K[np.ix_(idx, idx)]
            assert np.max(np.abs(block_12)) > 0.0

    def check_nonedge_zero_offdiagonals(K, elements, n_nodes):
        connected = {tuple(sorted((el['node_i'], el['node_j']))) for el in elements}
        for a in range(n_nodes):
            for b in range(n_nodes):
                if a == b:
                    continue
                if tuple(sorted((a, b))) not in connected:
                    block_ab = K[idx_block(a), idx_block(b)]
                    assert np.allclose(block_ab, 0.0, atol=1e-12, rtol=0.0)
    nodes_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_single = build_elements([(0, 1)])
    K_single = fcn(nodes_single, elements_single)
    check_basic_properties(K_single, nodes_single)
    check_element_contributions(K_single, elements_single)
    check_nonedge_zero_offdiagonals(K_single, elements_single, nodes_single.shape[0])
    nodes_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_chain = build_elements([(0, 1), (1, 2)])
    K_chain = fcn(nodes_chain, elements_chain)
    check_basic_properties(K_chain, nodes_chain)
    check_element_contributions(K_chain, elements_chain)
    check_nonedge_zero_offdiagonals(K_chain, elements_chain, nodes_chain.shape[0])
    nodes_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2.0, 0.0]])
    elements_triangle = build_elements([(0, 1), (1, 2), (2, 0)])
    K_triangle = fcn(nodes_triangle, elements_triangle)
    check_basic_properties(K_triangle, nodes_triangle)
    check_element_contributions(K_triangle, elements_triangle)
    nodes_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_square = build_elements([(0, 1), (1, 2), (2, 3), (3, 0)])
    K_square = fcn(nodes_square, elements_square)
    check_basic_properties(K_square, nodes_square)
    check_element_contributions(K_square, elements_square)
    check_nonedge_zero_offdiagonals(K_square, elements_square, nodes_square.shape[0])