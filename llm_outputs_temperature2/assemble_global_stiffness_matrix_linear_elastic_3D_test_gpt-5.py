def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location. Covers multiple
    structural configurations: single element, linear chain, triangle loop, and square loop.
    """

    def block(K, i, j):
        return K[i * 6:(i + 1) * 6, j * 6:(j + 1) * 6]
    tol_nonzero = 1e-10
    tol_zero = 1e-10
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 8.333e-06
    Iz = 8.333e-06
    J = 1.667e-05
    lz = [0.0, 0.0, 1.0]
    nodes_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    elements_single = [dict(node_i=0, node_j=1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=lz)]
    K_single = fcn(nodes_single, elements_single)
    n_single = nodes_single.shape[0]
    assert K_single.shape == (6 * n_single, 6 * n_single)
    assert np.allclose(K_single, K_single.T, rtol=1e-08, atol=1e-08)
    assert np.linalg.norm(block(K_single, 0, 0)) > tol_nonzero
    assert np.linalg.norm(block(K_single, 1, 1)) > tol_nonzero
    assert np.linalg.norm(block(K_single, 0, 1)) > tol_nonzero
    assert np.linalg.norm(block(K_single, 1, 0)) > tol_nonzero
    nodes_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements_chain = [dict(node_i=0, node_j=1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=lz), dict(node_i=1, node_j=2, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=lz)]
    edges_chain = {(0, 1), (1, 2)}
    K_chain = fcn(nodes_chain, elements_chain)
    n_chain = nodes_chain.shape[0]
    assert K_chain.shape == (6 * n_chain, 6 * n_chain)
    assert np.allclose(K_chain, K_chain.T, rtol=1e-08, atol=1e-08)
    for (i, j) in edges_chain:
        assert np.linalg.norm(block(K_chain, i, j)) > tol_nonzero
        assert np.linalg.norm(block(K_chain, j, i)) > tol_nonzero
    for i in range(n_chain):
        for j in range(i + 1, n_chain):
            if (i, j) not in edges_chain:
                assert np.linalg.norm(block(K_chain, i, j)) < tol_zero
                assert np.linalg.norm(block(K_chain, j, i)) < tol_zero
    nodes_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3.0) / 2.0, 0.0]], dtype=float)
    elements_triangle = [dict(node_i=0, node_j=1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=lz), dict(node_i=1, node_j=2, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=lz), dict(node_i=2, node_j=0, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=lz)]
    edges_triangle = {(0, 1), (1, 2), (0, 2)}
    K_triangle = fcn(nodes_triangle, elements_triangle)
    n_triangle = nodes_triangle.shape[0]
    assert K_triangle.shape == (6 * n_triangle, 6 * n_triangle)
    assert np.allclose(K_triangle, K_triangle.T, rtol=1e-08, atol=1e-08)
    for (i, j) in edges_triangle:
        assert np.linalg.norm(block(K_triangle, i, j)) > tol_nonzero
        assert np.linalg.norm(block(K_triangle, j, i)) > tol_nonzero
    nodes_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    elements_square = [dict(node_i=0, node_j=1, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=lz), dict(node_i=1, node_j=2, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=lz), dict(node_i=2, node_j=3, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=lz), dict(node_i=3, node_j=0, E=E, nu=nu, A=A, I_y=Iy, I_z=Iz, J=J, local_z=lz)]
    edges_square = {(0, 1), (1, 2), (2, 3), (0, 3)}
    K_square = fcn(nodes_square, elements_square)
    n_square = nodes_square.shape[0]
    assert K_square.shape == (6 * n_square, 6 * n_square)
    assert np.allclose(K_square, K_square.T, rtol=1e-08, atol=1e-08)
    for (i, j) in edges_square:
        assert np.linalg.norm(block(K_square, i, j)) > tol_nonzero
        assert np.linalg.norm(block(K_square, j, i)) > tol_nonzero
    non_edges_square = {(0, 2), (1, 3)}
    for (i, j) in non_edges_square:
        assert np.linalg.norm(block(K_square, i, j)) < tol_zero
        assert np.linalg.norm(block(K_square, j, i)) < tol_zero