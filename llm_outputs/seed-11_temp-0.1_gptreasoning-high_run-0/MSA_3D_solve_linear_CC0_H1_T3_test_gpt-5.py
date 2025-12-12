def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """

    def build_masks(n_nodes, bc):
        ndofs = 6 * n_nodes
        fixed = np.zeros(ndofs, dtype=bool)
        for (node_idx, mask) in bc.items():
            arr = np.asarray(mask, dtype=bool)
            start = 6 * node_idx
            fixed[start:start + 6] = arr
        free = ~fixed
        return (fixed, free)
    cases = [dict(n_nodes=2, seed=202, bc={0: np.ones(6, dtype=bool)}), dict(n_nodes=3, seed=303, bc={0: np.ones(6, dtype=bool)})]
    for case in cases:
        n_nodes = case['n_nodes']
        seed = case['seed']
        bc = case['bc']
        ndofs = 6 * n_nodes
        rng = np.random.default_rng(seed)
        A = rng.standard_normal((ndofs, ndofs))
        K_global = A.T @ A + 50.0 * np.eye(ndofs)
        (fixed, free) = build_masks(n_nodes, bc)
        P_global = np.zeros(ndofs)
        P_global[free] = rng.standard_normal(np.sum(free))
        (u, r) = fcn(P_global, K_global, bc, n_nodes)
        assert isinstance(u, np.ndarray) and isinstance(r, np.ndarray)
        assert u.shape == (ndofs,)
        assert r.shape == (ndofs,)
        K_ff = K_global[np.ix_(free, free)]
        K_sf = K_global[np.ix_(fixed, free)]
        P_f = P_global[free]
        P_s = P_global[fixed]
        u_f_expected = np.linalg.solve(K_ff, P_f)
        r_s_expected = K_sf @ u_f_expected - P_s
        assert np.allclose(u[fixed], 0.0, atol=1e-12, rtol=0.0)
        assert np.allclose(u[free], u_f_expected, rtol=1e-10, atol=1e-12)
        assert np.allclose(r[free], 0.0, atol=1e-12, rtol=0.0)
        assert np.allclose(r[fixed], r_s_expected, rtol=1e-10, atol=1e-12)
        assert np.allclose(K_ff @ u[free], P_f, rtol=1e-10, atol=1e-12)
        residual = K_global @ u - P_global
        assert np.allclose(residual, r, rtol=1e-10, atol=1e-12)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 1
    ndofs = 6
    bc = {0: np.array([True, False, False, False, False, False], dtype=bool)}
    diag_vals = np.array([1.0, 1.0, 1.0, 1e-20, 1.0, 1.0])
    K_global = np.diag(diag_vals)
    P_global = np.zeros(ndofs)
    with pytest.raises(ValueError):
        fcn(P_global, K_global, bc, n_nodes)