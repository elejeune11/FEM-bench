def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """

    def partition(boundary_conditions, n_nodes):
        fixed = []
        free = []
        for i in range(n_nodes):
            bc = boundary_conditions.get(i, [False] * 6)
            for j in range(6):
                idx = 6 * i + j
                if bc[j]:
                    fixed.append(idx)
                else:
                    free.append(idx)
        return (np.array(fixed, dtype=int), np.array(free, dtype=int))

    def verify_case(K, P, boundary, n_nodes):
        fixed, free = partition(boundary, n_nodes)
        u, r = fcn(P, K, boundary, n_nodes)
        assert u.shape == (6 * n_nodes,)
        assert r.shape == (6 * n_nodes,)
        if fixed.size > 0:
            assert np.allclose(u[fixed], 0.0, atol=1e-12, rtol=0.0)
        K_ff = K[np.ix_(free, free)]
        P_f = P[free]
        u_f_expected = np.linalg.solve(K_ff, P_f)
        assert np.allclose(u[free], u_f_expected, rtol=1e-10, atol=1e-12)
        r_expected = np.zeros_like(P)
        if fixed.size > 0:
            K_sf = K[np.ix_(fixed, free)]
            P_s = P[fixed]
            r_expected[fixed] = K_sf @ u_f_expected - P_s
        assert np.allclose(r, r_expected, rtol=1e-10, atol=1e-12)
        if free.size > 0:
            assert np.allclose(r[free], 0.0, atol=1e-12, rtol=0.0)
        residual = K @ u - P
        assert np.allclose(residual, r, rtol=1e-10, atol=1e-12)
    rng = np.random.default_rng(42)
    n_nodes = 2
    dof = 6 * n_nodes
    A = rng.normal(size=(dof, dof))
    K = A.T @ A + 50.0 * np.eye(dof)
    P = rng.normal(size=dof)
    boundary = {0: [True, True, True, True, True, True]}
    verify_case(K, P, boundary, n_nodes)
    rng2 = np.random.default_rng(123)
    A2 = rng2.normal(size=(dof, dof))
    K2 = A2.T @ A2 + 30.0 * np.eye(dof)
    P2 = rng2.normal(size=dof)
    boundary2 = {0: [True, True, True, False, False, False], 1: [False, False, False, True, True, False]}
    verify_case(K2, P2, boundary2, n_nodes)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 1
    dof = 6 * n_nodes
    K_global = np.zeros((dof, dof))
    P_global = np.zeros(dof)
    boundary_conditions = {0: [False, True, True, True, True, True]}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)