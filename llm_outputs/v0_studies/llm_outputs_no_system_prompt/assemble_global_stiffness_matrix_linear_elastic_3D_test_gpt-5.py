def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """

    def _local_elastic_stiffness_matrix_3D_beam(E, nu, A, L, I_y, I_z, J):
        G = E / (2.0 * (1.0 + nu))
        k_ax = E * A / L
        k_tor = G * J / L
        k = np.zeros((12, 12), dtype=float)
        for (a, b, s) in [(0, 0, 1.0), (0, 6, -1.0), (6, 0, -1.0), (6, 6, 1.0)]:
            k[a, b] += s * k_ax
        for (a, b, s) in [(3, 3, 1.0), (3, 9, -1.0), (9, 3, -1.0), (9, 9, 1.0)]:
            k[a, b] += s * k_tor
        return k

    def _beam_transformation_matrix_3D(xi, yi, zi, xj, yj, zj, local_z=None):
        pi = np.array([xi, yi, zi], dtype=float)
        pj = np.array([xj, yj, zj], dtype=float)
        v = pj - pi
        L = np.linalg.norm(v)
        if L == 0:
            R = np.eye(3)
        else:
            x_axis = v / L
            if local_z is not None:
                z_guess = np.array(local_z, dtype=float)
                nz = np.linalg.norm(z_guess)
                if nz == 0:
                    z_guess = np.array([0.0, 0.0, 1.0])
                else:
                    z_guess = z_guess / nz
            else:
                z_guess = np.array([0.0, 0.0, 1.0])
            if abs(np.dot(x_axis, z_guess)) > 0.99:
                z_guess = np.array([0.0, 1.0, 0.0])
            y_axis = np.cross(z_guess, x_axis)
            ny = np.linalg.norm(y_axis)
            if ny == 0:
                z_guess = np.array([1.0, 0.0, 0.0])
                y_axis = np.cross(z_guess, x_axis)
                ny = np.linalg.norm(y_axis)
            y_axis = y_axis / ny
            z_axis = np.cross(x_axis, y_axis)
            R = np.vstack((x_axis, y_axis, z_axis)).T
        T = np.zeros((12, 12), dtype=float)
        for i in range(4):
            T[3 * i:3 * i + 3, 3 * i:3 * i + 3] = R
        return T
    g = fcn.__globals__
    orig_local = g.get('local_elastic_stiffness_matrix_3D_beam', None)
    orig_trans = g.get('beam_transformation_matrix_3D', None)
    g['local_elastic_stiffness_matrix_3D_beam'] = _local_elastic_stiffness_matrix_3D_beam
    g['beam_transformation_matrix_3D'] = _beam_transformation_matrix_3D
    try:
        tol = 1e-12
        E = 210000000000.0
        nu = 0.3
        A = 0.01
        I_y = 1e-06
        I_z = 1e-06
        J = 2e-06

        def dof_slice(n):
            return slice(6 * n, 6 * n + 6)
        configs = []
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
        elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
        configs.append((node_coords, elements))
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
        configs.append((node_coords, elements))
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.0, 0.3]])
        elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
        configs.append((node_coords, elements))
        node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]])
        elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
        configs.append((node_coords, elements))
        for (node_coords, elements) in configs:
            n_nodes = node_coords.shape[0]
            K = fcn(node_coords, elements)
            assert isinstance(K, np.ndarray)
            assert K.shape == (6 * n_nodes, 6 * n_nodes)
            assert np.allclose(K, K.T, atol=1e-09, rtol=0.0)
            adj = set()
            for e in elements:
                i = int(e['node_i'])
                j = int(e['node_j'])
                adj.add(tuple(sorted((i, j))))
                idx = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
                sub = K[np.ix_(idx, idx)]
                assert np.any(np.abs(sub) > tol)
                kij = K[dof_slice(i), dof_slice(j)]
                assert np.any(np.abs(kij) > tol)
            for a in range(n_nodes):
                for b in range(a + 1, n_nodes):
                    if (a, b) not in adj:
                        block = K[dof_slice(a), dof_slice(b)]
                        assert np.all(np.abs(block) <= tol)
    finally:
        if orig_local is not None:
            g['local_elastic_stiffness_matrix_3D_beam'] = orig_local
        else:
            g.pop('local_elastic_stiffness_matrix_3D_beam', None)
        if orig_trans is not None:
            g['beam_transformation_matrix_3D'] = orig_trans
        else:
            g.pop('beam_transformation_matrix_3D', None)