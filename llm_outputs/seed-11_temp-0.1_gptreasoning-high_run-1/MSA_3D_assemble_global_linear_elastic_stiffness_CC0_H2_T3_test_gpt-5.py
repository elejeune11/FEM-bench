def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 1e-06
    Iz = 1e-06
    J = 2e-06
    z_dir = np.array([0.0, 0.0, 1.0])

    def element(i, j):
        return {'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': z_dir}

    def dof_slice(n):
        return slice(6 * n, 6 * n + 6)

    def block(K, i, j):
        return K[dof_slice(i), dof_slice(j)]

    def pair_indices(i, j):
        return list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))

    def nonzero_block(B, tol=1e-12):
        return np.max(np.abs(B)) > tol
    atol = 1e-10
    rtol = 1e-10
    configs = []
    nodes_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    elems_single = [element(0, 1)]
    connected_single = {(0, 1)}
    disconnected_single = set()
    configs.append((nodes_single, elems_single, connected_single, disconnected_single))
    nodes_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elems_chain = [element(0, 1), element(1, 2)]
    connected_chain = {(0, 1), (1, 2)}
    disconnected_chain = {(0, 2)}
    configs.append((nodes_chain, elems_chain, connected_chain, disconnected_chain))
    nodes_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0]], dtype=float)
    elems_triangle = [element(0, 1), element(1, 2), element(2, 0)]
    connected_triangle = {(0, 1), (1, 2), (0, 2)}
    disconnected_triangle = set()
    configs.append((nodes_triangle, elems_triangle, connected_triangle, disconnected_triangle))
    nodes_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    elems_square = [element(0, 1), element(1, 2), element(2, 3), element(3, 0)]
    connected_square = {(0, 1), (1, 2), (2, 3), (0, 3)}
    disconnected_square = {(0, 2), (1, 3)}
    configs.append((nodes_square, elems_square, connected_square, disconnected_square))
    for node_coords, elements, connected_pairs, disconnected_pairs in configs:
        n_nodes = node_coords.shape[0]
        K = fcn(node_coords, elements)
        assert isinstance(K, np.ndarray)
        assert K.shape == (6 * n_nodes, 6 * n_nodes)
        assert np.allclose(K, K.T, rtol=rtol, atol=atol)
        assert nonzero_block(K)
        for el in elements:
            i = el['node_i']
            j = el['node_j']
            idx = pair_indices(i, j)
            K_ij12 = K[np.ix_(idx, idx)]
            assert nonzero_block(K_ij12, tol=atol)
            K_ii = block(K, i, i)
            K_jj = block(K, j, j)
            K_ij = block(K, i, j)
            K_ji = block(K, j, i)
            assert nonzero_block(K_ii, tol=atol)
            assert nonzero_block(K_jj, tol=atol)
            assert nonzero_block(K_ij, tol=atol)
            assert nonzero_block(K_ji, tol=atol)
            assert np.allclose(K_ij, K_ji.T, rtol=rtol, atol=atol)
        for i, j in connected_pairs:
            Bij = block(K, i, j)
            assert nonzero_block(Bij, tol=atol)
        for i, j in disconnected_pairs:
            Bij = block(K, i, j)
            Bji = block(K, j, i)
            assert np.allclose(Bij, 0.0, atol=atol)
            assert np.allclose(Bji, 0.0, atol=atol)