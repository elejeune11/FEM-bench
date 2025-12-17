def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    The test covers multiple structural configurations:
    For each configuration, it checks:
    """
    import numpy as np

    def dof_slice(n):
        return slice(6 * n, 6 * n + 6)

    def check_case(node_coords, elements):
        K = fcn(node_coords, elements)
        n_nodes = node_coords.shape[0]
        assert K.shape == (6 * n_nodes, 6 * n_nodes)
        assert np.allclose(K, K.T, rtol=0.0, atol=1e-08)
        connected = set()
        for e in elements:
            i = int(e['node_i'])
            j = int(e['node_j'])
            connected.add(tuple(sorted((i, j))))
        for p in range(n_nodes):
            for q in range(p + 1, n_nodes):
                block = K[dof_slice(p), dof_slice(q)]
                if (p, q) in connected:
                    assert np.linalg.norm(block) > 1e-10
                else:
                    assert np.allclose(block, 0.0, atol=1e-12)
        for e in elements:
            i = int(e['node_i'])
            j = int(e['node_j'])
            idx = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
            sub = K[np.ix_(idx, idx)]
            assert np.linalg.norm(sub) > 1e-10
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 8.333e-06
    I_z = 8.333e-06
    J = 1.667e-05
    z_hat = np.array([0.0, 0.0, 1.0])
    nodes_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    elements_single = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': z_hat}]
    check_case(nodes_single, elements_single)
    nodes_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements_chain = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': z_hat}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    check_case(nodes_chain, elements_chain)
    nodes_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0]], dtype=float)
    elements_triangle = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': z_hat}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': z_hat}, {'node_i': 2, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': z_hat}]
    check_case(nodes_triangle, elements_triangle)
    nodes_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    elements_square = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': z_hat}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': z_hat}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': z_hat}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    check_case(nodes_square, elements_square)