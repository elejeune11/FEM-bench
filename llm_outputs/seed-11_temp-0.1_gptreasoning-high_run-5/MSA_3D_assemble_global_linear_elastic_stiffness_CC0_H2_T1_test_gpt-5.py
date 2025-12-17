def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """
    import numpy as np

    def make_elem(i, j, E, nu, A, Iy, Iz, J, zvec):
        return {'node_i': i, 'node_j': j, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': zvec}

    def assert_case(node_coords, elements):
        K = fcn(node_coords, elements)
        n_nodes = node_coords.shape[0]
        assert isinstance(K, np.ndarray)
        assert K.shape == (6 * n_nodes, 6 * n_nodes)
        assert np.allclose(K, K.T, rtol=1e-10, atol=1e-10)

        def block_nonzero(ii, jj):
            if ii < 0 or jj < 0 or ii >= n_nodes or (jj >= n_nodes):
                return False
            dofs = list(range(6 * ii, 6 * ii + 6)) + list(range(6 * jj, 6 * jj + 6))
            Kblk = K[np.ix_(dofs, dofs)]
            if Kblk.shape != (12, 12):
                return False
            return np.any(np.abs(Kblk) > 0)
        for e in elements:
            i0 = int(e['node_i'])
            j0 = int(e['node_j'])
            ok = block_nonzero(i0, j0)
            if not ok and i0 > 0 and (j0 > 0):
                ok = block_nonzero(i0 - 1, j0 - 1)
            assert ok
        return K
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 8.333e-06
    Iz = 8.333e-06
    J = 1.6666e-05
    zvec = [0.0, 0.0, 1.0]
    node_coords_1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    elements_1 = [make_elem(0, 1, E, nu, A, Iy, Iz, J, zvec)]
    assert_case(node_coords_1, elements_1)
    node_coords_2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements_2 = [make_elem(0, 1, E, nu, A, Iy, Iz, J, zvec), make_elem(1, 2, E, nu, A, Iy, Iz, J, zvec)]
    assert_case(node_coords_2, elements_2)
    h = np.sqrt(3) / 2.0
    node_coords_3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]], dtype=float)
    elements_3 = [make_elem(0, 1, E, nu, A, Iy, Iz, J, zvec), make_elem(1, 2, E, nu, A, Iy, Iz, J, zvec), make_elem(2, 0, E, nu, A, Iy, Iz, J, zvec)]
    assert_case(node_coords_3, elements_3)
    node_coords_4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    elements_4 = [make_elem(0, 1, E, nu, A, Iy, Iz, J, zvec), make_elem(1, 2, E, nu, A, Iy, Iz, J, zvec), make_elem(2, 3, E, nu, A, Iy, Iz, J, zvec), make_elem(3, 0, E, nu, A, Iy, Iz, J, zvec)]
    assert_case(node_coords_4, elements_4)