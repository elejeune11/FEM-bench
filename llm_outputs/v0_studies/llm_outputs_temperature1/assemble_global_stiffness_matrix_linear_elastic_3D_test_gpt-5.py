def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Configurations covered:
    """
    tol = 1e-09

    def dof_idx(n):
        return list(range(6 * n, 6 * n + 6))

    def build_elements(nodes, edges):
        elements = []
        E = 210000000000.0
        nu = 0.3
        A = 0.01
        I_y = 1e-06
        I_z = 1e-06
        J = 2e-06
        for (i, j) in edges:
            vi = np.array(nodes[j], dtype=float) - np.array(nodes[i], dtype=float)
            if np.linalg.norm(np.cross(vi, np.array([0.0, 0.0, 1.0]))) < 1e-12:
                local_z = [0.0, 1.0, 0.0]
            else:
                local_z = [0.0, 0.0, 1.0]
            elements.append({'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J, 'local_z': local_z})
        return elements

    def run_and_check(nodes, edges):
        node_coords = np.array(nodes, dtype=float)
        n_nodes = node_coords.shape[0]
        elements = build_elements(nodes, edges)
        K = fcn(node_coords, elements)
        assert K.shape == (6 * n_nodes, 6 * n_nodes)
        assert np.allclose(K, K.T, atol=1e-08)
        edge_set = {frozenset((i, j)) for (i, j) in edges}
        for (i, j) in edges:
            off_ij = K[np.ix_(dof_idx(i), dof_idx(j))]
            off_ji = K[np.ix_(dof_idx(j), dof_idx(i))]
            assert np.linalg.norm(off_ij) > tol
            assert np.linalg.norm(off_ji) > tol
        for (i, j) in combinations(range(n_nodes), 2):
            if frozenset((i, j)) not in edge_set:
                off_ij = K[np.ix_(dof_idx(i), dof_idx(j))]
                off_ji = K[np.ix_(dof_idx(j), dof_idx(i))]
                assert np.linalg.norm(off_ij) < 1e-08
                assert np.linalg.norm(off_ji) < 1e-08
        connected_nodes = set()
        for (i, j) in edges:
            connected_nodes.add(i)
            connected_nodes.add(j)
        for n in connected_nodes:
            diag = K[np.ix_(dof_idx(n), dof_idx(n))]
            assert np.linalg.norm(diag) > tol
        return K
    nodes_single = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0)]
    edges_single = [(0, 1)]
    run_and_check(nodes_single, edges_single)
    nodes_chain = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)]
    edges_chain = [(0, 1), (1, 2)]
    run_and_check(nodes_chain, edges_chain)
    nodes_triangle = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)]
    edges_triangle = [(0, 1), (1, 2), (2, 0)]
    run_and_check(nodes_triangle, edges_triangle)
    nodes_square = [(0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.0), (0.0, 1.0, 0.0)]
    edges_square = [(0, 1), (1, 2), (2, 3), (3, 0)]
    run_and_check(nodes_square, edges_square)