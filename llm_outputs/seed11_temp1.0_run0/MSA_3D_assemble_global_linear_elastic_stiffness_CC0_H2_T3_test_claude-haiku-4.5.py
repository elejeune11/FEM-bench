def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop."""
    node_coords_1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_1 = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-05, 'I_z': 8.33e-05, 'J': 0.000167, 'local_z': np.array([0.0, 0.0, 1.0])}]
    K_1 = fcn(node_coords_1, elements_1)
    assert K_1.shape == (12, 12), f'Expected shape (12, 12), got {K_1.shape}'
    assert np.allclose(K_1, K_1.T, atol=1e-10), 'Single element stiffness matrix is not symmetric'
    assert np.any(np.abs(K_1) > 1e-06), 'Single element stiffness matrix is all zeros'
    node_coords_2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_2 = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-05, 'I_z': 8.33e-05, 'J': 0.000167, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-05, 'I_z': 8.33e-05, 'J': 0.000167, 'local_z': np.array([0.0, 0.0, 1.0])}]
    K_2 = fcn(node_coords_2, elements_2)
    assert K_2.shape == (18, 18), f'Expected shape (18, 18), got {K_2.shape}'
    assert np.allclose(K_2, K_2.T, atol=1e-10), 'Linear chain stiffness matrix is not symmetric'
    assert np.any(np.abs(K_2) > 1e-06), 'Linear chain stiffness matrix is all zeros'
    middle_dof_block = K_2[6:12, 6:12]
    assert np.any(np.abs(middle_dof_block) > 1e-06), 'Middle node DOF block has no contributions'
    node_coords_3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2, 0.0]])
    elements_3 = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-05, 'I_z': 8.33e-05, 'J': 0.000167, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-05, 'I_z': 8.33e-05, 'J': 0.000167, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 2, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-05, 'I_z': 8.33e-05, 'J': 0.000167, 'local_z': np.array([0.0, 0.0, 1.0])}]
    K_3 = fcn(node_coords_3, elements_3)
    assert K_3.shape == (18, 18), f'Expected shape (18, 18), got {K_3.shape}'
    assert np.allclose(K_3, K_3.T, atol=1e-10), 'Triangle loop stiffness matrix is not symmetric'
    assert np.any(np.abs(K_3) > 1e-06), 'Triangle loop stiffness matrix is all zeros'
    node_coords_4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_4 = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-05, 'I_z': 8.33e-05, 'J': 0.000167, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-05, 'I_z': 8.33e-05, 'J': 0.000167, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 2, 'node_j': 3, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-05, 'I_z': 8.33e-05, 'J': 0.000167, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 3, 'node_j': 0, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.33e-05, 'I_z': 8.33e-05, 'J': 0.000167, 'local_z': np.array([0.0, 0.0, 1.0])}]
    K_4 = fcn(node_coords_4, elements_4)
    assert K_4.shape == (24, 24), f'Expected shape (24, 24), got {K_4.shape}'
    assert np.allclose(K_4, K_4.T, atol=1e-10), 'Square loop stiffness matrix is not symmetric'
    assert np.any(np.abs(K_4) > 1e-06), 'Square loop stiffness matrix is all zeros'
    for node_idx in range(4):
        node_dof_start = node_idx * 6
        node_dof_end = node_dof_start + 6
        node_dof_block = K_4[node_dof_start:node_dof_end, node_dof_start:node_dof_end]
        assert np.any(np.abs(node_dof_block) > 1e-06), f'Node {node_idx} DOF block has no contributions'