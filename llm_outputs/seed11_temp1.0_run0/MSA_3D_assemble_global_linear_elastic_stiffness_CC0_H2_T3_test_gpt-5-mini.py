def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """
    import numpy as np
    tol_nonzero = 1e-12
    tol_sym = 1e-08
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-06
    I_z = 2e-06
    J = 5e-07
    configs = []
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    configs.append((node_coords, elements))
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    configs.append((node_coords, elements))
    node_coords = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2.0, 0.0]])
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
        assert np.allclose(K, K.T, atol=tol_sym, rtol=1e-06)
        for el in elements:
            ni = int(el['node_i'])
            nj = int(el['node_j'])
            rows = list(range(6 * ni, 6 * ni + 6)) + list(range(6 * nj, 6 * nj + 6))
            cols = rows
            K_e = K[np.ix_(rows, cols)]
            assert K_e.shape == (12, 12)
            assert np.any(np.abs(K_e) > tol_nonzero)