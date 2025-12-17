def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    The test covers multiple structural configurations:
    """
    import numpy as np

    def dof_indices(node_index):
        return [6 * node_index + k for k in range(6)]

    def run_case(node_coords, edge_list):
        E = 210000000000.0
        nu = 0.3
        A = 0.01
        I_y = 1e-06
        I_z = 1e-06
        J = 2e-06
        node_coords = np.asarray(node_coords, dtype=float)
        elements = []
        for i, j in edge_list:
            xi, yi, zi = node_coords[i]
            xj, yj, zj = node_coords[j]
            v = np.array([xj - xi, yj - yi, zj - zi], dtype=float)
            v_norm = np.linalg.norm(v)
            assert v_norm > 0
            v_unit = v / v_norm
            lz = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(v_unit, lz)) > 0.999:
                lz = np.array([0.0, 1.0, 0.0])
            elements.append({'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': lz})
        K = fcn(node_coords, elements)
        n_nodes = node_coords.shape[0]
        assert K.shape == (6 * n_nodes, 6 * n_nodes)
        assert np.allclose(K, K.T, atol=1e-08, rtol=1e-08)
        for i, j in edge_list:
            idx_i = dof_indices(i)
            idx_j = dof_indices(j)
            block_idx = idx_i + idx_j
            sub_12x12 = K[np.ix_(block_idx, block_idx)]
            assert np.linalg.norm(sub_12x12, ord=np.inf) > 0.0
            off_block = K[np.ix_(idx_i, idx_j)]
            assert np.linalg.norm(off_block, ord=np.inf) > 0.0
        connected = set(((min(i, j), max(i, j)) for i, j in edge_list))
        for i in range(n_nodes):
            for j in range(i + 1, n_nodes):
                if (i, j) not in connected:
                    block = K[np.ix_(dof_indices(i), dof_indices(j))]
                    assert np.allclose(block, 0.0, atol=1e-09)
    run_case(node_coords=[(0.0, 0.0, 0.0), (1.5, 0.0, 0.0)], edge_list=[(0, 1)])
    run_case(node_coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.2, 0.0, 0.0)], edge_list=[(0, 1), (1, 2)])
    sqrt3 = np.sqrt(3.0)
    run_case(node_coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.5, sqrt3 / 2.0, 0.0)], edge_list=[(0, 1), (1, 2), (2, 0)])
    run_case(node_coords=[(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)], edge_list=[(0, 1), (1, 2), (2, 3), (3, 0)])