def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """
    atol = 1e-10
    rtol = 1e-08

    def dof_indices_for_node(node_index):
        start = 6 * node_index
        return list(range(start, start + 6))

    def submatrix_for_element(K, ni, nj):
        dofs = dof_indices_for_node(ni) + dof_indices_for_node(nj)
        return K[np.ix_(dofs, dofs)]

    def check_scenario(node_coords, elements):
        n_nodes = node_coords.shape[0]
        K = fcn(node_coords, elements)
        assert isinstance(K, np.ndarray)
        assert K.shape == (6 * n_nodes, 6 * n_nodes)
        assert np.allclose(K, K.T, rtol=rtol, atol=atol)
        for el in elements:
            ni = el['node_i']
            nj = el['node_j']
            sub = submatrix_for_element(K, ni, nj)
            assert np.any(np.abs(sub) > 1e-12)
        return K
    node_coords_1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements_1 = [{'node_i': 0, 'node_j': 1, 'E': 210000000000.0, 'nu': 0.3, 'A': 0.01, 'I_y': 8.333e-06, 'I_z': 8.333e-06, 'J': 1.6666e-05, 'local_z': np.array([0.0, 0.0, 1.0])}]
    K1 = check_scenario(node_coords_1, elements_1)
    node_coords_2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements_2 = [{'node_i': 0, 'node_j': 1, 'E': 70000000000.0, 'nu': 0.29, 'A': 0.008, 'I_y': 6e-06, 'I_z': 6e-06, 'J': 1.2e-05, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': 70000000000.0, 'nu': 0.29, 'A': 0.008, 'I_y': 6e-06, 'I_z': 6e-06, 'J': 1.2e-05, 'local_z': np.array([0.0, 0.0, 1.0])}]
    K2 = check_scenario(node_coords_2, elements_2)
    dofs_0 = dof_indices_for_node(0)
    dofs_2 = dof_indices_for_node(2)
    sub_0_2 = K2[np.ix_(dofs_0, dofs_2)]
    assert np.allclose(sub_0_2, 0.0, atol=1e-12, rtol=0.0)
    node_coords_3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    elements_3 = [{'node_i': 0, 'node_j': 1, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.009, 'I_y': 7e-06, 'I_z': 7e-06, 'J': 1.3e-05, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.009, 'I_y': 7e-06, 'I_z': 7e-06, 'J': 1.3e-05, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 2, 'node_j': 0, 'E': 200000000000.0, 'nu': 0.3, 'A': 0.009, 'I_y': 7e-06, 'I_z': 7e-06, 'J': 1.3e-05, 'local_z': np.array([0.0, 0.0, 1.0])}]
    K3 = check_scenario(node_coords_3, elements_3)
    node_coords_4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
    elements_4 = [{'node_i': 0, 'node_j': 1, 'E': 190000000000.0, 'nu': 0.28, 'A': 0.01, 'I_y': 8e-06, 'I_z': 8e-06, 'J': 1.5e-05, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 1, 'node_j': 2, 'E': 190000000000.0, 'nu': 0.28, 'A': 0.01, 'I_y': 8e-06, 'I_z': 8e-06, 'J': 1.5e-05, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 2, 'node_j': 3, 'E': 190000000000.0, 'nu': 0.28, 'A': 0.01, 'I_y': 8e-06, 'I_z': 8e-06, 'J': 1.5e-05, 'local_z': np.array([0.0, 0.0, 1.0])}, {'node_i': 3, 'node_j': 0, 'E': 190000000000.0, 'nu': 0.28, 'A': 0.01, 'I_y': 8e-06, 'I_z': 8e-06, 'J': 1.5e-05, 'local_z': np.array([0.0, 0.0, 1.0])}]
    K4 = check_scenario(node_coords_4, elements_4)