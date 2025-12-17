def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.02
    Iy = 2.5e-06
    Iz = 3.1e-06
    J = 1.2e-06

    def make_elements(edges):
        return [{'node_i': int(i), 'node_j': int(j), 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J} for i, j in edges]

    def node_dofs(n):
        return np.arange(6 * n, 6 * n + 6)

    def check_structure(node_coords, edges):
        elements = make_elements(edges)
        K = fcn(np.asarray(node_coords, dtype=float), elements)
        K = np.asarray(K, dtype=float)
        n = node_coords.shape[0]
        assert K.shape == (6 * n, 6 * n)
        assert np.allclose(K, K.T, atol=1e-08, rtol=1e-08)
        for i, j in edges:
            idx = np.concatenate([node_dofs(i), node_dofs(j)])
            K_sub_12 = K[np.ix_(idx, idx)]
            assert np.linalg.norm(K_sub_12) > 1e-12
            K_ij = K[np.ix_(node_dofs(i), node_dofs(j))]
            K_ji = K[np.ix_(node_dofs(j), node_dofs(i))]
            assert np.linalg.norm(K_ij) > 1e-12
            assert np.linalg.norm(K_ji) > 1e-12
            assert np.allclose(K_ij.T, K_ji, atol=1e-08, rtol=1e-08)
        connected = {(min(i, j), max(i, j)) for i, j in edges}
        for a in range(n):
            for b in range(n):
                if a == b:
                    continue
                if (min(a, b), max(a, b)) not in connected:
                    off_block = K[np.ix_(node_dofs(a), node_dofs(b))]
                    assert np.linalg.norm(off_block) <= 1e-12
    coords_single = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=float)
    edges_single = [(0, 1)]
    check_structure(coords_single, edges_single)
    coords_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    edges_chain = [(0, 1), (1, 2)]
    check_structure(coords_chain, edges_chain)
    coords_triangle = np.array([[0.0, 0.0, 0.0], [1.2, 0.0, 0.0], [0.2, 0.8, 0.7]], dtype=float)
    edges_triangle = [(0, 1), (1, 2), (2, 0)]
    check_structure(coords_triangle, edges_triangle)
    coords_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    edges_square = [(0, 1), (1, 2), (2, 3), (3, 0)]
    check_structure(coords_square, edges_square)