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

    def make_element(node_i, node_j, local_z=None):
        elem = {'node_i': node_i, 'node_j': node_j, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}
        if local_z is not None:
            elem['local_z'] = local_z
        return elem
    node_coords_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_single = [make_element(0, 1)]
    K_single = fcn(node_coords_single, elements_single)
    n_nodes_single = 2
    expected_shape_single = (6 * n_nodes_single, 6 * n_nodes_single)
    assert K_single.shape == expected_shape_single, f'Single element: Expected shape {expected_shape_single}, got {K_single.shape}'
    assert np.allclose(K_single, K_single.T, rtol=1e-10, atol=1e-10), 'Single element: Global stiffness matrix is not symmetric'
    block_single = K_single[:12, :12]
    assert np.any(np.abs(block_single) > 0), 'Single element: Element did not contribute nonzero values to its block'
    node_coords_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_chain = [make_element(0, 1), make_element(1, 2)]
    K_chain = fcn(node_coords_chain, elements_chain)
    n_nodes_chain = 3
    expected_shape_chain = (6 * n_nodes_chain, 6 * n_nodes_chain)
    assert K_chain.shape == expected_shape_chain, f'Linear chain: Expected shape {expected_shape_chain}, got {K_chain.shape}'
    assert np.allclose(K_chain, K_chain.T, rtol=1e-10, atol=1e-10), 'Linear chain: Global stiffness matrix is not symmetric'
    block_01_ii = K_chain[0:6, 0:6]
    block_01_jj = K_chain[6:12, 6:12]
    assert np.any(np.abs(block_01_ii) > 0), 'Linear chain: Element 0-1 did not contribute to node 0 block'
    assert np.any(np.abs(block_01_jj) > 0), 'Linear chain: Element 0-1 did not contribute to node 1 block'
    block_12_jj = K_chain[12:18, 12:18]
    assert np.any(np.abs(block_12_jj) > 0), 'Linear chain: Element 1-2 did not contribute to node 2 block'
    node_coords_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    elements_triangle = [make_element(0, 1), make_element(1, 2), make_element(2, 0)]
    K_triangle = fcn(node_coords_triangle, elements_triangle)
    n_nodes_triangle = 3
    expected_shape_triangle = (6 * n_nodes_triangle, 6 * n_nodes_triangle)
    assert K_triangle.shape == expected_shape_triangle, f'Triangle loop: Expected shape {expected_shape_triangle}, got {K_triangle.shape}'
    assert np.allclose(K_triangle, K_triangle.T, rtol=1e-10, atol=1e-10), 'Triangle loop: Global stiffness matrix is not symmetric'
    for i in range(3):
        block_diag = K_triangle[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6]
        assert np.any(np.abs(block_diag) > 0), f'Triangle loop: Node {i} diagonal block is zero'
    node_coords_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_square = [make_element(0, 1), make_element(1, 2), make_element(2, 3), make_element(3, 0)]
    K_square = fcn(node_coords_square, elements_square)
    n_nodes_square = 4
    expected_shape_square = (6 * n_nodes_square, 6 * n_nodes_square)
    assert K_square.shape == expected_shape_square, f'Square loop: Expected shape {expected_shape_square}, got {K_square.shape}'
    assert np.allclose(K_square, K_square.T, rtol=1e-10, atol=1e-10), 'Square loop: Global stiffness matrix is not symmetric'
    for i in range(4):
        block_diag = K_square[i * 6:(i + 1) * 6, i * 6:(i + 1) * 6]
        assert np.any(np.abs(block_diag) > 0), f'Square loop: Node {i} diagonal block is zero'
    block_01 = K_square[0:6, 6:12]
    assert np.any(np.abs(block_01) > 0), 'Square loop: Off-diagonal block between nodes 0 and 1 is zero'
    block_12 = K_square[6:12, 12:18]
    assert np.any(np.abs(block_12) > 0), 'Square loop: Off-diagonal block between nodes 1 and 2 is zero'
    node_coords_3d = np.array([[0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    elements_3d = [make_element(0, 1)]
    K_3d = fcn(node_coords_3d, elements_3d)
    n_nodes_3d = 2
    expected_shape_3d = (6 * n_nodes_3d, 6 * n_nodes_3d)
    assert K_3d.shape == expected_shape_3d, f'3D z-axis element: Expected shape {expected_shape_3d}, got {K_3d.shape}'
    assert np.allclose(K_3d, K_3d.T, rtol=1e-10, atol=1e-10), '3D z-axis element: Global stiffness matrix is not symmetric'
    node_coords_L = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0]])
    elements_L = [make_element(0, 1), make_element(1, 2)]
    K_L = fcn(node_coords_L, elements_L)
    n_nodes_L = 3
    expected_shape_L = (6 * n_nodes_L, 6 * n_nodes_L)
    assert K_L.shape == expected_shape_L, f'3D L-frame: Expected shape {expected_shape_L}, got {K_L.shape}'
    assert np.allclose(K_L, K_L.T, rtol=1e-10, atol=1e-10), '3D L-frame: Global stiffness matrix is not symmetric'
    block_middle = K_L[6:12, 6:12]
    assert np.any(np.abs(block_middle) > 0), '3D L-frame: Middle node diagonal block is zero'
    node_coords_orient = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    local_z_custom = np.array([0.0, 1.0, 0.0])
    elements_orient = [make_element(0, 1, local_z=local_z_custom)]
    K_orient = fcn(node_coords_orient, elements_orient)
    assert K_orient.shape == (12, 12), f'Custom orientation: Expected shape (12, 12), got {K_orient.shape}'
    assert np.allclose(K_orient, K_orient.T, rtol=1e-10, atol=1e-10), 'Custom orientation: Global stiffness matrix is not symmetric'