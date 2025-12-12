def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes_a = 2
    dof_per_node = 6
    N_a = dof_per_node * n_nodes_a
    A = 2.0 * np.eye(dof_per_node)
    C = 4.0 * np.eye(dof_per_node)
    B = 0.5 * np.eye(dof_per_node) + 0.2 * np.ones((dof_per_node, dof_per_node))
    K_a = np.block([[A, B], [B, C]])
    P_a = np.zeros(N_a)
    P_f_a = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    P_a[dof_per_node:] = P_f_a
    bc_a = {0: np.array([True, True, True, True, True, True], dtype=bool)}
    (u_a, r_a) = fcn(P_a, K_a, bc_a, n_nodes_a)
    u_free_exp_a = np.linalg.solve(C, P_f_a)
    u_exp_a = np.zeros(N_a)
    u_exp_a[dof_per_node:] = u_free_exp_a
    r_exp_a = np.zeros(N_a)
    r_exp_a[:dof_per_node] = B @ u_free_exp_a
    assert u_a.shape == (N_a,)
    assert r_a.shape == (N_a,)
    assert np.allclose(u_a, u_exp_a, rtol=1e-12, atol=1e-12)
    assert np.allclose(r_a, r_exp_a, rtol=1e-12, atol=1e-12)
    K_ff_a = C
    assert np.allclose(K_ff_a @ u_a[dof_per_node:], P_f_a, rtol=1e-12, atol=1e-12)
    assert np.allclose(K_a @ u_a + r_a, P_a, rtol=1e-12, atol=1e-12)
    assert np.allclose(r_a[dof_per_node:], 0.0, rtol=1e-12, atol=1e-12)
    n_nodes_b = 1
    N_b = dof_per_node * n_nodes_b
    L = np.zeros((dof_per_node, dof_per_node))
    diag_vals = [2.0, 2.1, 1.9, 2.2, 2.05, 1.8]
    for (i, v) in enumerate(diag_vals):
        L[i, i] = v
    L[0, 1] = 0.3
    L[1, 2] = 0.25
    L[2, 3] = 0.2
    L[3, 4] = 0.15
    L[4, 5] = 0.1
    K_b = L.T @ L
    P_b = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0], dtype=float)
    fixed_mask_b = np.array([True, False, True, False, True, False], dtype=bool)
    bc_b = {0: fixed_mask_b}
    free_mask_b = ~fixed_mask_b
    K_ff_b = K_b[np.ix_(free_mask_b, free_mask_b)]
    K_sf_b = K_b[np.ix_(fixed_mask_b, free_mask_b)]
    P_f_b = P_b[free_mask_b]
    P_s_b = P_b[fixed_mask_b]
    u_free_exp_b = np.linalg.solve(K_ff_b, P_f_b)
    u_exp_b = np.zeros(N_b)
    u_exp_b[free_mask_b] = u_free_exp_b
    r_exp_b = np.zeros(N_b)
    r_exp_b[fixed_mask_b] = K_sf_b @ u_free_exp_b - P_s_b
    (u_b, r_b) = fcn(P_b, K_b, bc_b, n_nodes_b)
    assert u_b.shape == (N_b,)
    assert r_b.shape == (N_b,)
    assert np.allclose(u_b, u_exp_b, rtol=1e-12, atol=1e-12)
    assert np.allclose(r_b, r_exp_b, rtol=1e-12, atol=1e-12)
    assert np.allclose(K_ff_b @ u_b[free_mask_b], P_f_b, rtol=1e-12, atol=1e-12)
    assert np.allclose(K_b @ u_b + r_b, P_b, rtol=1e-12, atol=1e-12)
    assert np.allclose(r_b[free_mask_b], 0.0, rtol=1e-12, atol=1e-12)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 1
    dof_per_node = 6
    N = dof_per_node * n_nodes
    diag = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 0.0], dtype=float)
    K = np.diag(diag)
    P = np.zeros(N, dtype=float)
    bc = {0: np.array([True, True, True, True, True, False], dtype=bool)}
    with pytest.raises(ValueError):
        fcn(P, K, bc, n_nodes)