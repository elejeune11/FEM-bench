def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """
    import numpy as np

    def dof_slice(n):
        return slice(6 * n, 6 * n + 6)

    def block(K, i, j):
        return K[dof_slice(i), dof_slice(j)]
    props = dict(E=210000000000.0, nu=0.3, A=0.01, I_y=8.333e-06, I_z=8.333e-06, J=1.667e-05)
    coords1 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    elements1 = [{'node_i': 0, 'node_j': 1, 'local_z': np.array([0.0, 0.0, 1.0]), **props}]
    K1 = fcn(coords1, elements1)
    n1 = coords1.shape[0]
    assert K1.shape == (6 * n1, 6 * n1)
    assert np.allclose(K1, K1.T, rtol=1e-10, atol=1e-10)
    tol1 = 1e-12 * max(1.0, np.linalg.norm(K1, ord=np.inf))
    assert np.linalg.norm(block(K1, 0, 0), ord=np.inf) > tol1
    assert np.linalg.norm(block(K1, 1, 1), ord=np.inf) > tol1
    assert np.linalg.norm(block(K1, 0, 1), ord=np.inf) > tol1
    assert np.linalg.norm(block(K1, 1, 0), ord=np.inf) > tol1
    coords2 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements2 = [{'node_i': 0, 'node_j': 1, 'local_z': np.array([0.0, 0.0, 1.0]), **props}, {'node_i': 1, 'node_j': 2, 'local_z': np.array([0.0, 0.0, 1.0]), **props}]
    K2 = fcn(coords2, elements2)
    n2 = coords2.shape[0]
    assert K2.shape == (6 * n2, 6 * n2)
    assert np.allclose(K2, K2.T, rtol=1e-10, atol=1e-10)
    tol2 = 1e-12 * max(1.0, np.linalg.norm(K2, ord=np.inf))
    for node in range(n2):
        assert np.linalg.norm(block(K2, node, node), ord=np.inf) > tol2
    assert np.linalg.norm(block(K2, 0, 1), ord=np.inf) > tol2
    assert np.linalg.norm(block(K2, 1, 0), ord=np.inf) > tol2
    assert np.linalg.norm(block(K2, 1, 2), ord=np.inf) > tol2
    assert np.linalg.norm(block(K2, 2, 1), ord=np.inf) > tol2
    assert np.linalg.norm(block(K2, 0, 2), ord=np.inf) <= tol2
    assert np.linalg.norm(block(K2, 2, 0), ord=np.inf) <= tol2
    coords3 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, np.sqrt(3) / 2.0, 0.0]], dtype=float)
    elements3 = [{'node_i': 0, 'node_j': 1, 'local_z': np.array([0.0, 0.0, 1.0]), **props}, {'node_i': 1, 'node_j': 2, 'local_z': np.array([0.0, 0.0, 1.0]), **props}, {'node_i': 2, 'node_j': 0, 'local_z': np.array([0.0, 0.0, 1.0]), **props}]
    K3 = fcn(coords3, elements3)
    n3 = coords3.shape[0]
    assert K3.shape == (6 * n3, 6 * n3)
    assert np.allclose(K3, K3.T, rtol=1e-10, atol=1e-10)
    tol3 = 1e-12 * max(1.0, np.linalg.norm(K3, ord=np.inf))
    for node in range(n3):
        assert np.linalg.norm(block(K3, node, node), ord=np.inf) > tol3
    connected_pairs_tri = [(0, 1), (1, 2), (2, 0)]
    for (i, j) in connected_pairs_tri:
        assert np.linalg.norm(block(K3, i, j), ord=np.inf) > tol3
        assert np.linalg.norm(block(K3, j, i), ord=np.inf) > tol3
    coords4 = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    elements4 = [{'node_i': 0, 'node_j': 1, 'local_z': np.array([0.0, 0.0, 1.0]), **props}, {'node_i': 1, 'node_j': 2, 'local_z': np.array([0.0, 0.0, 1.0]), **props}, {'node_i': 2, 'node_j': 3, 'local_z': np.array([0.0, 0.0, 1.0]), **props}, {'node_i': 3, 'node_j': 0, 'local_z': np.array([0.0, 0.0, 1.0]), **props}]
    K4 = fcn(coords4, elements4)
    n4 = coords4.shape[0]
    assert K4.shape == (6 * n4, 6 * n4)
    assert np.allclose(K4, K4.T, rtol=1e-10, atol=1e-10)
    tol4 = 1e-12 * max(1.0, np.linalg.norm(K4, ord=np.inf))
    for node in range(n4):
        assert np.linalg.norm(block(K4, node, node), ord=np.inf) > tol4
    connected_pairs_sq = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for (i, j) in connected_pairs_sq:
        assert np.linalg.norm(block(K4, i, j), ord=np.inf) > tol4
        assert np.linalg.norm(block(K4, j, i), ord=np.inf) > tol4
    non_connected_pairs_sq = [(0, 2), (1, 3)]
    for (i, j) in non_connected_pairs_sq:
        assert np.linalg.norm(block(K4, i, j), ord=np.inf) <= tol4
        assert np.linalg.norm(block(K4, j, i), ord=np.inf) <= tol4