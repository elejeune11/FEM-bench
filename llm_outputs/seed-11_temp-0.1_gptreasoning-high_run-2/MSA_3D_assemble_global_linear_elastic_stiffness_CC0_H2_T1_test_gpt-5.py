def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    The test covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """
    import numpy as np
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 1e-06
    Iz = 1e-06
    J = 2e-06

    def make_elem(i, j, local_z=None):
        el = {'node_i': int(i), 'node_j': int(j), 'E': float(E), 'nu': float(nu), 'A': float(A), 'I_y': float(Iy), 'I_z': float(Iz), 'J': float(J)}
        if local_z is not None:
            el['local_z'] = np.asarray(local_z, dtype=float)
        return el
    z_axis = np.array([0.0, 0.0, 1.0], dtype=float)
    h = float(np.sqrt(3.0) / 2.0)
    cases = []
    nodes_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    elems_single = [make_elem(0, 1, z_axis)]
    cases.append((nodes_single, elems_single))
    nodes_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elems_chain = [make_elem(0, 1, z_axis), make_elem(1, 2, z_axis)]
    cases.append((nodes_chain, elems_chain))
    nodes_triangle = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, h, 0.0]], dtype=float)
    elems_triangle = [make_elem(0, 1, z_axis), make_elem(1, 2, z_axis), make_elem(2, 0, z_axis)]
    cases.append((nodes_triangle, elems_triangle))
    nodes_square = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    elems_square = [make_elem(0, 1, z_axis), make_elem(1, 2, z_axis), make_elem(2, 3, z_axis), make_elem(3, 0, z_axis)]
    cases.append((nodes_square, elems_square))
    for node_coords, elements in cases:
        K = fcn(node_coords, elements)
        n_nodes = node_coords.shape[0]
        assert isinstance(K, np.ndarray)
        assert K.shape == (6 * n_nodes, 6 * n_nodes)
        assert np.allclose(K, K.T, rtol=1e-09, atol=1e-09)
        for el in elements:
            i = int(el['node_i'])
            j = int(el['node_j'])
            dofs = np.concatenate([np.arange(6 * i, 6 * i + 6), np.arange(6 * j, 6 * j + 6)])
            subK = K[np.ix_(dofs, dofs)]
            assert subK.shape == (12, 12)
            assert np.any(np.abs(subK) > 0.0)