def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 2
    dof_per_node = 6
    size = dof_per_node * n_nodes
    boundary_conditions = {0: np.array([True, True, True, True, True, True], dtype=bool)}
    diag_node0 = np.array([20.0, 25.0, 30.0, 35.0, 40.0, 45.0])
    diag_node1 = np.array([3.0, 10.0, 1.5, 7.0, 4.0, 5.0])
    K_e_diag = np.concatenate([diag_node0, diag_node1])
    K_e_global = np.diag(K_e_diag)
    K_g_global = -np.eye(size)
    lam, vec = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    fixed_mask = np.zeros(size, dtype=bool)
    for node in range(n_nodes):
        bc_vec = np.array(boundary_conditions.get(node, np.zeros(dof_per_node, dtype=bool)), dtype=bool)
        fixed_mask[node * dof_per_node:(node + 1) * dof_per_node] = bc_vec
    free_mask = ~fixed_mask
    expected_lambda = float(np.min(diag_node1))
    assert abs(lam - expected_lambda) < 1e-10
    assert vec.shape == (size,)
    assert np.allclose(vec[fixed_mask], 0.0, atol=1e-12)
    phi_ff = vec[free_mask]
    assert np.any(np.abs(phi_ff) > 0.0)
    K_e_ff = K_e_global[np.ix_(free_mask, free_mask)]
    K_g_ff = K_g_global[np.ix_(free_mask, free_mask)]
    res = K_e_ff @ phi_ff + lam * K_g_ff @ phi_ff
    scale = max(1.0, np.linalg.norm(K_e_ff) + abs(lam) * np.linalg.norm(K_g_ff))
    assert np.linalg.norm(res) <= 1e-08 * scale
    expected_idx_free = int(np.argmin(diag_node1))
    dominant_idx = int(np.argmax(np.abs(phi_ff)))
    assert dominant_idx == expected_idx_free
    others = np.delete(phi_ff, dominant_idx)
    if np.abs(phi_ff[dominant_idx]) > 0:
        assert np.linalg.norm(others) <= 1e-06 * abs(phi_ff[dominant_idx]) + 1e-12

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 1
    dof_per_node = 6
    size = dof_per_node * n_nodes
    boundary_conditions = {0: np.array([False, True, True, True, True, True], dtype=bool)}
    K_e_diag = np.array([0.0, 10.0, 10.0, 10.0, 10.0, 10.0])
    K_e_global = np.diag(K_e_diag)
    K_g_global = -np.eye(size)
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    dof_per_node = 6
    size = dof_per_node * n_nodes
    boundary_conditions = {0: np.array([False, False, True, True, True, True], dtype=bool)}
    K_e_global = np.eye(size)
    K_g_global = -np.eye(size)
    J = np.array([[0.0, 1.0], [-1.0, 0.0]])
    K_g_global[0:2, 0:2] = J
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    dof_per_node = 6
    size = dof_per_node * n_nodes
    boundary_conditions = {0: np.array([False, False, True, True, True, True], dtype=bool)}
    K_e_diag = np.array([-3.0, -2.0, 10.0, 10.0, 10.0, 10.0])
    K_e_global = np.diag(K_e_diag)
    K_g_global = -np.eye(size)
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 1
    dof_per_node = 6
    size = dof_per_node * n_nodes
    boundary_conditions = {0: np.array([False, False, False, False, True, True], dtype=bool)}
    K_e_diag = np.array([2.0, 5.0, 8.0, 4.0, 100.0, 100.0])
    K_e_global = np.diag(K_e_diag)
    K_g_global = -np.eye(size)
    lam1, vec1 = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    c = 7.3
    lam2, vec2 = fcn(K_e_global, c * K_g_global, boundary_conditions, n_nodes)
    assert vec1.shape == (size,)
    assert vec2.shape == (size,)
    fixed_mask = np.array(boundary_conditions[0], dtype=bool)
    assert np.allclose(vec1[fixed_mask], 0.0, atol=1e-12)
    assert np.allclose(vec2[fixed_mask], 0.0, atol=1e-12)
    assert abs(lam2 - lam1 / c) <= 1e-10
    free_mask = ~fixed_mask
    phi_ff = vec2[free_mask]
    K_e_ff = K_e_global[np.ix_(free_mask, free_mask)]
    K_g_ff = (c * K_g_global)[np.ix_(free_mask, free_mask)]
    res = K_e_ff @ phi_ff + lam2 * K_g_ff @ phi_ff
    scale = max(1.0, np.linalg.norm(K_e_ff) + abs(lam2) * np.linalg.norm(K_g_ff))
    assert np.linalg.norm(res) <= 1e-08 * scale