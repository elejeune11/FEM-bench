def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations, for example: single element, linear chain, triangle loop, and square loop.
    """
    import numpy as np

    def check_case(node_coords, elements):
        K = np.asarray(fcn(np.asarray(node_coords, dtype=float), elements))
        n_nodes = len(node_coords)
        assert K.shape == (6 * n_nodes, 6 * n_nodes)
        assert np.allclose(K, K.T, rtol=1e-06, atol=1e-09)
        for el in elements:
            i = int(el['node_i'])
            j = int(el['node_j'])
            rows = list(range(6 * i, 6 * i + 6)) + list(range(6 * j, 6 * j + 6))
            cols = rows
            sub = K[np.ix_(rows, cols)]
            assert np.linalg.norm(sub) > 0
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    I_y = 1e-06
    I_z = 1e-06
    J = 1e-06
    nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    check_case(nodes, elements)
    nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    check_case(nodes, elements)
    nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 0.8660254037844386, 0.0]]
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    check_case(nodes, elements)
    nodes = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]]
    elements = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': I_y, 'I_z': I_z, 'J': J}]
    check_case(nodes, elements)