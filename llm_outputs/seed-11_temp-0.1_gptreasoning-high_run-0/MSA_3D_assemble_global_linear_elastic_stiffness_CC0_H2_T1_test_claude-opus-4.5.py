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
    expected_size_single = 6 * n_nodes_single
    assert K_single.shape == (expected_size_single, expected_size_single), f'Single element: Expected shape ({expected_size_single}, {expected_size_single}), got {K_single.shape}'
    assert np.allclose(K_single, K_single.T, rtol=1e-10, atol=1e-10), 'Single element: Global stiffness matrix is not symmetric'
    assert np.any(K_single != 0), 'Single element: Global stiffness matrix should have nonzero entries'
    node_coords_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_chain = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    K_chain = fcn(node_coords_chain, elements_chain)
    n_nodes_chain = 3
    expected_size_chain = 6 * n_nodes_chain
    assert K_chain.shape == (expected_size_chain, expected_size_chain), f'Linear chain: Expected shape ({expected_size_chain}, {expected_size_chain}), got {K_chain.shape}'
    assert np.allclose(K_chain, K_chain.T, rtol=1e-10, atol=1e-10), 'Linear chain: Global stiffness matrix is not symmetric'
    block_node1 = K_chain[6:12, 6:12]
    assert np.any(block_node1 != 0), 'Linear chain: Middle node should have nonzero stiffness contributions'
    node_coords_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    elements_triangle = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    K_triangle = fcn(node_coords_triangle, elements_triangle)
    n_nodes_triangle = 3
    expected_size_triangle = 6 * n_nodes_triangle
    assert K_triangle.shape == (expected_size_triangle, expected_size_triangle), f'Triangle loop: Expected shape ({expected_size_triangle}, {expected_size_triangle}), got {K_triangle.shape}'
    assert np.allclose(K_triangle, K_triangle.T, rtol=1e-10, atol=1e-10), 'Triangle loop: Global stiffness matrix is not symmetric'
    for node_idx in range(3):
        start_dof = node_idx * 6
        end_dof = start_dof + 6
        block = K_triangle[start_dof:end_dof, start_dof:end_dof]
        assert np.any(block != 0), f'Triangle loop: Node {node_idx} should have nonzero diagonal block'
    node_coords_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_square = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    K_square = fcn(node_coords_square, elements_square)
    n_nodes_square = 4
    expected_size_square = 6 * n_nodes_square
    assert K_square.shape == (expected_size_square, expected_size_square), f'Square loop: Expected shape ({expected_size_square}, {expected_size_square}), got {K_square.shape}'
    assert np.allclose(K_square, K_square.T, rtol=1e-10, atol=1e-10), 'Square loop: Global stiffness matrix is not symmetric'
    for node_idx in range(4):
        start_dof = node_idx * 6
        end_dof = start_dof + 6
        block = K_square[start_dof:end_dof, start_dof:end_dof]
        assert np.any(block != 0), f'Square loop: Node {node_idx} should have nonzero diagonal block'
    node_coords_tetra = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0], [0.5, np.sqrt(3) / 6, np.sqrt(2 / 3)]])
    elements_tetra = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 0, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    K_tetra = fcn(node_coords_tetra, elements_tetra)
    n_nodes_tetra = 4
    expected_size_tetra = 6 * n_nodes_tetra
    assert K_tetra.shape == (expected_size_tetra, expected_size_tetra), f'Tetrahedron: Expected shape ({expected_size_tetra}, {expected_size_tetra}), got {K_tetra.shape}'
    assert np.allclose(K_tetra, K_tetra.T, rtol=1e-10, atol=1e-10), 'Tetrahedron: Global stiffness matrix is not symmetric'
    for node_idx in range(4):
        start_dof = node_idx * 6
        end_dof = start_dof + 6
        block = K_tetra[start_dof:end_dof, start_dof:end_dof]
        assert np.any(block != 0), f'Tetrahedron: Node {node_idx} should have nonzero diagonal block'
    block_01 = K_tetra[0:6, 6:12]
    assert np.any(block_01 != 0), 'Tetrahedron: Off-diagonal block for connected nodes 0-1 should be nonzero'