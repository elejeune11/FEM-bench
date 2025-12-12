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
    node_coords_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_single = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    K_single = fcn(node_coords_single, elements_single)
    n_nodes_single = 2
    expected_shape_single = (6 * n_nodes_single, 6 * n_nodes_single)
    assert K_single.shape == expected_shape_single, f'Single element: Expected shape {expected_shape_single}, got {K_single.shape}'
    assert np.allclose(K_single, K_single.T, atol=1e-10), 'Single element: Global stiffness matrix is not symmetric'
    block_00 = K_single[0:6, 0:6]
    block_01 = K_single[0:6, 6:12]
    block_11 = K_single[6:12, 6:12]
    assert np.linalg.norm(block_00) > 0, 'Single element: Block (0,0) should be nonzero'
    assert np.linalg.norm(block_01) > 0, 'Single element: Block (0,1) should be nonzero'
    assert np.linalg.norm(block_11) > 0, 'Single element: Block (1,1) should be nonzero'
    node_coords_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_chain = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    K_chain = fcn(node_coords_chain, elements_chain)
    n_nodes_chain = 3
    expected_shape_chain = (6 * n_nodes_chain, 6 * n_nodes_chain)
    assert K_chain.shape == expected_shape_chain, f'Linear chain: Expected shape {expected_shape_chain}, got {K_chain.shape}'
    assert np.allclose(K_chain, K_chain.T, atol=1e-10), 'Linear chain: Global stiffness matrix is not symmetric'
    assert np.linalg.norm(K_chain[0:6, 0:6]) > 0, 'Linear chain: Block (0,0) should be nonzero'
    assert np.linalg.norm(K_chain[6:12, 6:12]) > 0, 'Linear chain: Block (1,1) should be nonzero'
    assert np.linalg.norm(K_chain[12:18, 12:18]) > 0, 'Linear chain: Block (2,2) should be nonzero'
    node_coords_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    elements_triangle = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    K_triangle = fcn(node_coords_triangle, elements_triangle)
    n_nodes_triangle = 3
    expected_shape_triangle = (6 * n_nodes_triangle, 6 * n_nodes_triangle)
    assert K_triangle.shape == expected_shape_triangle, f'Triangle loop: Expected shape {expected_shape_triangle}, got {K_triangle.shape}'
    assert np.allclose(K_triangle, K_triangle.T, atol=1e-10), 'Triangle loop: Global stiffness matrix is not symmetric'
    for i in range(3):
        block_ii = K_triangle[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6]
        assert np.linalg.norm(block_ii) > 0, f'Triangle loop: Block ({i},{i}) should be nonzero'
    node_coords_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_square = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    K_square = fcn(node_coords_square, elements_square)
    n_nodes_square = 4
    expected_shape_square = (6 * n_nodes_square, 6 * n_nodes_square)
    assert K_square.shape == expected_shape_square, f'Square loop: Expected shape {expected_shape_square}, got {K_square.shape}'
    assert np.allclose(K_square, K_square.T, atol=1e-10), 'Square loop: Global stiffness matrix is not symmetric'
    for i in range(4):
        block_ii = K_square[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6]
        assert np.linalg.norm(block_ii) > 0, f'Square loop: Block ({i},{i}) should be nonzero'
    assert np.linalg.norm(K_square[0:6, 6:12]) > 0, 'Square loop: Block (0,1) should be nonzero'
    assert np.linalg.norm(K_square[0:6, 18:24]) > 0, 'Square loop: Block (0,3) should be nonzero'
    node_coords_3d = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    elements_3d = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 0, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 0, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    K_3d = fcn(node_coords_3d, elements_3d)
    n_nodes_3d = 4
    expected_shape_3d = (6 * n_nodes_3d, 6 * n_nodes_3d)
    assert K_3d.shape == expected_shape_3d, f'3D structure: Expected shape {expected_shape_3d}, got {K_3d.shape}'
    assert np.allclose(K_3d, K_3d.T, atol=1e-10), '3D structure: Global stiffness matrix is not symmetric'
    block_00_3d = K_3d[0:6, 0:6]
    assert np.linalg.norm(block_00_3d) > 0, '3D structure: Block (0,0) should be nonzero'
    assert np.linalg.norm(K_3d[0:6, 6:12]) > 0, '3D structure: Block (0,1) should be nonzero'
    assert np.linalg.norm(K_3d[0:6, 12:18]) > 0, '3D structure: Block (0,2) should be nonzero'
    assert np.linalg.norm(K_3d[0:6, 18:24]) > 0, '3D structure: Block (0,3) should be nonzero'
    assert np.allclose(K_3d[6:12, 12:18], 0, atol=1e-10), '3D structure: Block (1,2) should be zero'
    assert np.allclose(K_3d[6:12, 18:24], 0, atol=1e-10), '3D structure: Block (1,3) should be zero'
    assert np.allclose(K_3d[12:18, 18:24], 0, atol=1e-10), '3D structure: Block (2,3) should be zero'