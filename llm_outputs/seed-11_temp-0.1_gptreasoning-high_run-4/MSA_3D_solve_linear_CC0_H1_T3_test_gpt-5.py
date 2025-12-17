def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies linear_solve against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """

    def partition_from_bc(n_nodes, boundary_conditions):
        N = 6 * n_nodes
        fixed_mask = np.zeros(N, dtype=bool)
        for node_idx, dof_mask in boundary_conditions.items():
            m = np.array(dof_mask, dtype=bool)
            assert m.shape == (6,)
            start = 6 * node_idx
            fixed_mask[start:start + 6] = m
        free_mask = ~fixed_mask
        return (fixed_mask, free_mask)

    def run_case(n_nodes, bc, K_global, P_global):
        fixed_mask, free_mask = partition_from_bc(n_nodes, bc)
        u, r = fcn(P_global, K_global, bc, n_nodes)
        N = 6 * n_nodes
        assert isinstance(u, np.ndarray) and isinstance(r, np.ndarray)
        assert u.shape == (N,) and r.shape == (N,)
        assert np.allclose(u[fixed_mask], 0.0)
        assert np.allclose(r[free_mask], 0.0)
        K_ff = K_global[np.ix_(free_mask, free_mask)]
        P_f = P_global[free_mask]
        assert np.allclose(K_ff @ u[free_mask], P_f, rtol=1e-10, atol=1e-10)
        K_sf = K_global[np.ix_(fixed_mask, free_mask)]
        P_s = P_global[fixed_mask]
        r_expected_s = K_sf @ u[free_mask] - P_s
        assert np.allclose(r[fixed_mask], r_expected_s, rtol=1e-10, atol=1e-10)
        Ku = K_global @ u
        Pr = P_global + r
        assert np.allclose(Ku, Pr, rtol=1e-10, atol=1e-10)
    n_nodes_a = 2
    N_a = 6 * n_nodes_a
    A_a = np.tril(np.ones((N_a, N_a), dtype=float))
    K_a = A_a.T @ A_a + 0.5 * np.eye(N_a)
    P_a = np.arange(1, N_a + 1, dtype=float)
    bc_a = {0: np.array([True, True, True, True, True, True], dtype=bool)}
    run_case(n_nodes_a, bc_a, K_a, P_a)
    n_nodes_b = 3
    N_b = 6 * n_nodes_b
    B_b = np.eye(N_b) * 3.0 + np.tril(np.ones((N_b, N_b), dtype=float), k=-1) * 0.5
    K_b = B_b.T @ B_b + 0.25 * np.eye(N_b)
    P_b = (-1.0) ** np.arange(N_b) * (np.arange(N_b, dtype=float) + 1.0) / 3.0
    bc_b = {0: np.array([True, True, True, True, True, True], dtype=bool), 1: np.array([False, False, False, True, True, True], dtype=bool)}
    run_case(n_nodes_b, bc_b, K_b, P_b)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 1
    N = 6 * n_nodes
    K_global = np.zeros((N, N), dtype=float)
    P_global = np.zeros(N, dtype=float)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)