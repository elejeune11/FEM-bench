def test_assemble_global_stiffness_matrix_shape_and_symmetry(fcn):
    """
    Tests that the global stiffness matrix assembly function produces a symmetric matrix of correct shape,
    and that each element contributes a nonzero 12x12 block to the appropriate location.
    Covers multiple structural configurations: single element, linear chain, triangle loop, and square loop.
    """
    import numpy as np
    tol = 1e-09

    def dofs(n):
        return [6 * n + k for k in range(6)]

    def block_sum_abs(K, rows, cols):
        s = 0.0
        for i in rows:
            for j in cols:
                s += abs(float(K[i, j]))
        return s

    def assert_symmetric(K):
        n = K.shape[0]
        for i in range(n):
            for j in range(i + 1, n):
                assert abs(float(K[i, j] - K[j, i])) <= tol
    E = 210000000000.0
    nu = 0.3
    A = 0.01
    Iy = 1e-06
    Iz = 1e-06
    J = 2e-06
    lz = [0.0, 0.0, 1.0]
    nodes_single = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float)
    elements_single = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': lz}]
    K_single = fcn(nodes_single, elements_single)
    assert K_single.shape == (6 * len(nodes_single), 6 * len(nodes_single))
    assert_symmetric(K_single)
    (i0, i1) = (dofs(0), dofs(1))
    assert block_sum_abs(K_single, i0 + i1, i0 + i1) > 0.0
    assert block_sum_abs(K_single, i0, i0) > 0.0
    assert block_sum_abs(K_single, i1, i1) > 0.0
    assert block_sum_abs(K_single, i0, i1) > 0.0
    assert block_sum_abs(K_single, i1, i0) > 0.0
    nodes_chain = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]], dtype=float)
    elements_chain = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': lz}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': lz}]
    K_chain = fcn(nodes_chain, elements_chain)
    assert K_chain.shape == (6 * len(nodes_chain), 6 * len(nodes_chain))
    assert_symmetric(K_chain)
    (d0, d1, d2) = (dofs(0), dofs(1), dofs(2))
    assert block_sum_abs(K_chain, d0 + d1, d0 + d1) > 0.0
    assert block_sum_abs(K_chain, d1 + d2, d1 + d2) > 0.0
    assert block_sum_abs(K_chain, d0, d0) > 0.0
    assert block_sum_abs(K_chain, d1, d1) > 0.0
    assert block_sum_abs(K_chain, d2, d2) > 0.0
    assert block_sum_abs(K_chain, d0, d1) > 0.0
    assert block_sum_abs(K_chain, d1, d2) > 0.0
    assert block_sum_abs(K_chain, d0, d2) == 0.0
    assert block_sum_abs(K_chain, d2, d0) == 0.0
    nodes_tri = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.4, 0.8, 0.0]], dtype=float)
    elements_tri = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': lz}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': lz}, {'node_i': 2, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': lz}]
    K_tri = fcn(nodes_tri, elements_tri)
    assert K_tri.shape == (6 * len(nodes_tri), 6 * len(nodes_tri))
    assert_symmetric(K_tri)
    (t0, t1, t2) = (dofs(0), dofs(1), dofs(2))
    assert block_sum_abs(K_tri, t0 + t1, t0 + t1) > 0.0
    assert block_sum_abs(K_tri, t1 + t2, t1 + t2) > 0.0
    assert block_sum_abs(K_tri, t2 + t0, t2 + t0) > 0.0
    assert block_sum_abs(K_tri, t0, t1) > 0.0
    assert block_sum_abs(K_tri, t1, t2) > 0.0
    assert block_sum_abs(K_tri, t2, t0) > 0.0
    nodes_sq = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=float)
    elements_sq = [{'node_i': 0, 'node_j': 1, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': lz}, {'node_i': 1, 'node_j': 2, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': lz}, {'node_i': 2, 'node_j': 3, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': lz}, {'node_i': 3, 'node_j': 0, 'E': E, 'nu': nu, 'A': A, 'I_y': Iy, 'I_z': Iz, 'J': J, 'local_z': lz}]
    K_sq = fcn(nodes_sq, elements_sq)
    assert K_sq.shape == (6 * len(nodes_sq), 6 * len(nodes_sq))
    assert_symmetric(K_sq)
    (s0, s1, s2, s3) = (dofs(0), dofs(1), dofs(2), dofs(3))
    assert block_sum_abs(K_sq, s0 + s1, s0 + s1) > 0.0
    assert block_sum_abs(K_sq, s1 + s2, s1 + s2) > 0.0
    assert block_sum_abs(K_sq, s2 + s3, s2 + s3) > 0.0
    assert block_sum_abs(K_sq, s3 + s0, s3 + s0) > 0.0
    assert block_sum_abs(K_sq, s0, s1) > 0.0
    assert block_sum_abs(K_sq, s1, s2) > 0.0
    assert block_sum_abs(K_sq, s2, s3) > 0.0
    assert block_sum_abs(K_sq, s3, s0) > 0.0
    assert block_sum_abs(K_sq, s0, s2) == 0.0
    assert block_sum_abs(K_sq, s2, s0) == 0.0
    assert block_sum_abs(K_sq, s1, s3) == 0.0
    assert block_sum_abs(K_sq, s3, s1) == 0.0