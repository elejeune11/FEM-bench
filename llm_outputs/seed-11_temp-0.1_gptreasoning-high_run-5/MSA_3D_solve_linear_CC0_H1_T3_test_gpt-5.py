def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """

    def build_spd(n, seed):
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((n, n))
        return A.T @ A + 10.0 * np.eye(n)
    n_nodes_1 = 2
    dof_1 = 6 * n_nodes_1
    K1 = build_spd(dof_1, seed=42)
    rng1 = np.random.default_rng(100)
    P1 = rng1.standard_normal(dof_1)
    bc1 = {0: np.array([True, True, True, True, True, True], dtype=bool)}
    u1, r1 = fcn(P1, K1, bc1, n_nodes_1)
    fixed_mask_1 = np.zeros(dof_1, dtype=bool)
    for i in range(n_nodes_1):
        node_bc = bc1.get(i, np.zeros(6, dtype=bool))
        fixed_mask_1[6 * i:6 * (i + 1)] = node_bc
    free_mask_1 = ~fixed_mask_1
    free_idx_1 = np.flatnonzero(free_mask_1)
    fixed_idx_1 = np.flatnonzero(fixed_mask_1)
    K_ff_1 = K1[np.ix_(free_idx_1, free_idx_1)]
    K_sf_1 = K1[np.ix_(fixed_idx_1, free_idx_1)]
    P_f_1 = P1[free_idx_1]
    P_s_1 = P1[fixed_idx_1]
    assert u1.shape == (dof_1,)
    assert r1.shape == (dof_1,)
    assert np.allclose(u1[fixed_idx_1], 0.0, atol=1e-12)
    u_f_expected_1 = np.linalg.solve(K_ff_1, P_f_1)
    assert np.allclose(u1[free_idx_1], u_f_expected_1, rtol=1e-10, atol=1e-12)
    r_expected_1 = np.zeros_like(P1)
    r_expected_1[fixed_idx_1] = K_sf_1 @ u_f_expected_1 - P_s_1
    assert np.allclose(r1, r_expected_1, rtol=1e-10, atol=1e-12)
    assert np.allclose(K1 @ u1, P1 + r1, rtol=1e-10, atol=1e-12)
    assert np.allclose(K_ff_1 @ u1[free_idx_1], P_f_1, rtol=1e-10, atol=1e-12)
    assert np.allclose(r1[free_idx_1], 0.0, atol=1e-12)
    n_nodes_2 = 2
    dof_2 = 6 * n_nodes_2
    K2 = build_spd(dof_2, seed=7)
    rng2 = np.random.default_rng(200)
    P2 = rng2.standard_normal(dof_2)
    bc2 = {0: np.array([True, True, True, False, False, False], dtype=bool)}
    u2, r2 = fcn(P2, K2, bc2, n_nodes_2)
    fixed_mask_2 = np.zeros(dof_2, dtype=bool)
    for i in range(n_nodes_2):
        node_bc = bc2.get(i, np.zeros(6, dtype=bool))
        fixed_mask_2[6 * i:6 * (i + 1)] = node_bc
    free_mask_2 = ~fixed_mask_2
    free_idx_2 = np.flatnonzero(free_mask_2)
    fixed_idx_2 = np.flatnonzero(fixed_mask_2)
    K_ff_2 = K2[np.ix_(free_idx_2, free_idx_2)]
    K_sf_2 = K2[np.ix_(fixed_idx_2, free_idx_2)]
    P_f_2 = P2[free_idx_2]
    P_s_2 = P2[fixed_idx_2]
    assert u2.shape == (dof_2,)
    assert r2.shape == (dof_2,)
    assert np.allclose(u2[fixed_idx_2], 0.0, atol=1e-12)
    u_f_expected_2 = np.linalg.solve(K_ff_2, P_f_2)
    assert np.allclose(u2[free_idx_2], u_f_expected_2, rtol=1e-10, atol=1e-12)
    r_expected_2 = np.zeros_like(P2)
    r_expected_2[fixed_idx_2] = K_sf_2 @ u_f_expected_2 - P_s_2
    assert np.allclose(r2, r_expected_2, rtol=1e-10, atol=1e-12)
    assert np.allclose(K2 @ u2, P2 + r2, rtol=1e-10, atol=1e-12)
    assert np.allclose(K_ff_2 @ u2[free_idx_2], P_f_2, rtol=1e-10, atol=1e-12)
    assert np.allclose(r2[free_idx_2], 0.0, atol=1e-12)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 1
    dof = 6 * n_nodes
    diag_vals = np.array([1.0, 1e-17, 2.0, 3.0, 4.0, 5.0])
    K = np.diag(diag_vals)
    bc = {0: np.array([False, False, True, True, True, True], dtype=bool)}
    P = np.array([1.0, -2.0, 0.0, 0.0, 0.0, 0.0], dtype=float)
    with pytest.raises(ValueError):
        fcn(P, K, bc, n_nodes)