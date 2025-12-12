def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 0.0001
    I_z = 0.0001
    J = 0.0002

    def create_element(node_i, node_j, local_z=None):
        elem = {'node_i': node_i, 'node_j': node_j, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}
        if local_z is not None:
            elem['local_z'] = local_z
        return elem
    node_coords_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_single = [create_element(0, 1)]
    K_single = fcn(node_coords_single, elements_single)
    n_nodes_single = 2
    expected_size_single = 6 * n_nodes_single
    assert K_single.shape == (expected_size_single, expected_size_single), f'Single element: Expected shape ({expected_size_single}, {expected_size_single}), got {K_single.shape}'
    assert np.allclose(K_single, K_single.T, rtol=1e-10, atol=1e-10), 'Single element: Global stiffness matrix is not symmetric'
    block_single = K_single[0:12, 0:12]
    assert np.any(np.abs(block_single) > 1e-20), 'Single element: 12x12 block should be nonzero'
    node_coords_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_chain = [create_element(0, 1), create_element(1, 2)]
    K_chain = fcn(node_coords_chain, elements_chain)
    n_nodes_chain = 3
    expected_size_chain = 6 * n_nodes_chain
    assert K_chain.shape == (expected_size_chain, expected_size_chain), f'Linear chain: Expected shape ({expected_size_chain}, {expected_size_chain}), got {K_chain.shape}'
    assert np.allclose(K_chain, K_chain.T, rtol=1e-10, atol=1e-10), 'Linear chain: Global stiffness matrix is not symmetric'
    block_node1 = K_chain[6:12, 6:12]
    assert np.any(np.abs(block_node1) > 1e-20), 'Linear chain: Middle node block should be nonzero'
    block_01 = K_chain[0:6, 6:12]
    block_12 = K_chain[6:12, 12:18]
    assert np.any(np.abs(block_01) > 1e-20), 'Linear chain: Coupling block between nodes 0-1 should be nonzero'
    assert np.any(np.abs(block_12) > 1e-20), 'Linear chain: Coupling block between nodes 1-2 should be nonzero'
    block_02 = K_chain[0:6, 12:18]
    assert np.allclose(block_02, 0, atol=1e-20), 'Linear chain: Nodes 0 and 2 should not be directly coupled'
    node_coords_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    elements_triangle = [create_element(0, 1), create_element(1, 2), create_element(2, 0)]
    K_triangle = fcn(node_coords_triangle, elements_triangle)
    n_nodes_triangle = 3
    expected_size_triangle = 6 * n_nodes_triangle
    assert K_triangle.shape == (expected_size_triangle, expected_size_triangle), f'Triangle: Expected shape ({expected_size_triangle}, {expected_size_triangle}), got {K_triangle.shape}'
    assert np.allclose(K_triangle, K_triangle.T, rtol=1e-10, atol=1e-10), 'Triangle: Global stiffness matrix is not symmetric'
    for i in range(3):
        block_diag = K_triangle[6 * i:6 * (i + 1), 6 * i:6 * (i + 1)]
        assert np.any(np.abs(block_diag) > 1e-20), f'Triangle: Diagonal block for node {i} should be nonzero'
    block_01_tri = K_triangle[0:6, 6:12]
    block_12_tri = K_triangle[6:12, 12:18]
    block_02_tri = K_triangle[0:6, 12:18]
    assert np.any(np.abs(block_01_tri) > 1e-20), 'Triangle: Coupling block 0-1 should be nonzero'
    assert np.any(np.abs(block_12_tri) > 1e-20), 'Triangle: Coupling block 1-2 should be nonzero'
    assert np.any(np.abs(block_02_tri) > 1e-20), 'Triangle: Coupling block 0-2 should be nonzero'
    node_coords_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_square = [create_element(0, 1), create_element(1, 2), create_element(2, 3), create_element(3, 0)]
    K_square = fcn(node_coords_square, elements_square)
    n_nodes_square = 4
    expected_size_square = 6 * n_nodes_square
    assert K_square.shape == (expected_size_square, expected_size_square), f'Square: Expected shape ({expected_size_square}, {expected_size_square}), got {K_square.shape}'
    assert np.allclose(K_square, K_square.T, rtol=1e-10, atol=1e-10), 'Square: Global stiffness matrix is not symmetric'
    for i in range(4):
        block_diag = K_square[6 * i:6 * (i + 1), 6 * i:6 * (i + 1)]
        assert np.any(np.abs(block_diag) > 1e-20), f'Square: Diagonal block for node {i} should be nonzero'
    block_01_sq = K_square[0:6, 6:12]
    block_12_sq = K_square[6:12, 12:18]
    block_23_sq = K_square[12:18, 18:24]
    block_30_sq = K_square[18:24, 0:6]
    assert np.any(np.abs(block_01_sq) > 1e-20), 'Square: Coupling block 0-1 should be nonzero'
    assert np.any(np.abs(block_12_sq) > 1e-20), 'Square: Coupling block 1-2 should be nonzero'
    assert np.any(np.abs(block_23_sq) > 1e-20), 'Square: Coupling block 2-3 should be nonzero'
    assert np.any(np.abs(block_30_sq) > 1e-20), 'Square: Coupling block 3-0 should be nonzero'
    block_02_sq = K_square[0:6, 12:18]
    block_13_sq = K_square[6:12, 18:24]
    assert np.allclose(block_02_sq, 0, atol=1e-20), 'Square: Non-adjacent nodes 0-2 should not be directly coupled'
    assert np.allclose(block_13_sq, 0, atol=1e-20), 'Square: Non-adjacent nodes 1-3 should not be directly coupled'
    node_coords_z = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    elements_z = [create_element(0, 1)]
    K_z = fcn(node_coords_z, elements_z)
    n_nodes_z = 2
    expected_size_z = 6 * n_nodes_z
    assert K_z.shape == (expected_size_z, expected_size_z), f'Z-axis element: Expected shape ({expected_size_z}, {expected_size_z}), got {K_z.shape}'
    assert np.allclose(K_z, K_z.T, rtol=1e-10, atol=1e-10), 'Z-axis element: Global stiffness matrix is not symmetric'
    block_z = K_z[0:12, 0:12]
    assert np.any(np.abs(block_z) > 1e-20), 'Z-axis element: 12x12 block should be nonzero'
    node_coords_diag = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    elements_diag = [create_element(0, 1)]
    K_diag = fcn(node_coords_diag, elements_diag)
    n_nodes_diag = 2
    expected_size_diag = 6 * n_nodes_diag
    assert K_diag.shape == (expected_size_diag, expected_size_diag), f'Diagonal element: Expected shape ({expected_size_diag}, {expected_size_diag}), got {K_diag.shape}'
    assert np.allclose(K_diag, K_diag.T, rtol=1e-10, atol=1e-10), 'Diagonal element: Global stiffness matrix is not symmetric'
    block_diag_elem = K_diag[0:12, 0:12]
    assert np.any(np.abs(block_diag_elem) > 1e-20), 'Diagonal element: 12x12 block should be nonzero'