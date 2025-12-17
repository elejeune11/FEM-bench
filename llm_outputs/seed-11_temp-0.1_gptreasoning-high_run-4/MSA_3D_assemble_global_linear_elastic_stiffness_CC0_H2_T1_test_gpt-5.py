def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """
    import numpy as np
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 1e-06
    Iz = 1e-06
    J = 2e-06
    local_z = np.array([0.0, 0.0, 1.0])

    def dof_indices(node):
        return list(range(6 * node, 6 * node + 6))

    def offdiag_block(K, i, j):
        ri = dof_indices(i)
        rj = dof_indices(j)
        return K[np.ix_(ri, rj)]

    def pair_block_12x12(K, i, j):
        idx = dof_indices(i) + dof_indices(j)
        return K[np.ix_(idx, idx)]

    def build_elements(edges):
        return [{'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': local_z} for i, j in edges]

    def assemble_and_basic_checks(node_coords, edges):
        elements = build_elements(edges)
        n = node_coords.shape[0]
        K = fcn(node_coords, elements)
        assert isinstance(K, np.ndarray)
        assert K.shape == (6 * n, 6 * n)
        assert np.allclose(K, K.T, atol=1e-08, rtol=1e-08)
        scale = np.linalg.norm(K, ord='fro')
        assert scale > 0.0
        for i, j in edges:
            blk12 = pair_block_12x12(K, i, j)
            off6 = offdiag_block(K, i, j)
            assert np.linalg.norm(blk12, ord='fro') > 1e-12 * scale
            assert np.linalg.norm(off6, ord='fro') > 1e-12 * scale
        edge_set = {tuple(sorted((i, j))) for i, j in edges}
        for i in range(node_coords.shape[0]):
            for j in range(node_coords.shape[0]):
                if j <= i:
                    continue
                if tuple(sorted((i, j))) not in edge_set:
                    off6 = offdiag_block(K, i, j)
                    assert np.allclose(off6, 0.0, atol=max(1e-12 * scale, 1e-14))
        return K
    nodes_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    edges_single = [(0, 1)]
    assemble_and_basic_checks(nodes_single, edges_single)
    nodes_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    edges_chain = [(0, 1), (1, 2)]
    K_chain = assemble_and_basic_checks(nodes_chain, edges_chain)
    off_0_2 = offdiag_block(K_chain, 0, 2)
    scale_chain = np.linalg.norm(K_chain, ord='fro')
    assert np.allclose(off_0_2, 0.0, atol=max(1e-12 * scale_chain, 1e-14))
    nodes_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2.0, 0.0]], dtype=float)
    edges_triangle = [(0, 1), (1, 2), (2, 0)]
    assemble_and_basic_checks(nodes_triangle, edges_triangle)
    nodes_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    edges_square = [(0, 1), (1, 2), (2, 3), (3, 0)]
    K_square = assemble_and_basic_checks(nodes_square, edges_square)
    off_0_2_sq = offdiag_block(K_square, 0, 2)
    off_1_3_sq = offdiag_block(K_square, 1, 3)
    scale_square = np.linalg.norm(K_square, ord='fro')
    assert np.allclose(off_0_2_sq, 0.0, atol=max(1e-12 * scale_square, 1e-14))
    assert np.allclose(off_1_3_sq, 0.0, atol=max(1e-12 * scale_square, 1e-14))