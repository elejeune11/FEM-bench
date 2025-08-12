def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location."""
    import numpy as np
    node_coords_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_single = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    K_single = fcn(node_coords_single, elements_single)
    assert K_single.shape == (12, 12)
    assert np.allclose(K_single, K_single.T)
    node_coords_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_chain = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    K_chain = fcn(node_coords_chain, elements_chain)
    assert K_chain.shape == (18, 18)
    assert np.allclose(K_chain, K_chain.T)
    node_coords_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.866, 0.0]])
    elements_triangle = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 0.0001, 'I_z': 0.0001, 'J': 0.0002}]
    K_triangle = fcn(node_coords_triangle, elements_triangle)
    assert K_triangle.shape == (18, 18)
    assert np.allclose(K_triangle, K_triangle.T)
    for (elements, K) in [(elements_single, K_single), (elements_chain, K_chain), (elements_triangle, K_triangle)]:
        for element in elements:
            (i, j) = (element['node_i'], element['node_j'])
            idx_i = slice(6 * i, 6 * (i + 1))
            idx_j = slice(6 * j, 6 * (j + 1))
            assert not np.allclose(K[idx_i, idx_i], 0)
            assert not np.allclose(K[idx_j, idx_j], 0)
            assert not np.allclose(K[idx_i, idx_j], 0)
            assert not np.allclose(K[idx_j, idx_i], 0)