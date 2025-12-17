def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """

    def build_spd(n, seed, diag_min=10.0, diag_max=30.0, offdiag_scale=0.001):
        rng = np.random.default_rng(seed)
        D = rng.uniform(diag_min, diag_max, size=n)
        K = np.diag(D)
        R = rng.uniform(-1.0, 1.0, size=(n, n))
        R = (R + R.T) / 2.0
        np.fill_diagonal(R, 0.0)
        K += offdiag_scale * R
        return K

    def partition_indices(boundary_conditions, n_nodes):
        fixed = []
        for node in range(n_nodes):
            bc = boundary_conditions.get(node, [False] * 6)
            for dof in range(6):
                if bool(bc[dof]):
                    fixed.append(6 * node + dof)
        fixed = np.array(sorted(fixed), dtype=int)
        all_idx = np.arange(6 * n_nodes, dtype=int)
        if fixed.size == 0:
            free = all_idx
        else:
            free = np.setdiff1d(all_idx, fixed, assume_unique=False)
        return (fixed, free)
    n_nodes1 = 2
    ndofs1 = 6 * n_nodes1
    K1 = build_spd(ndofs1, seed=42, diag_min=15.0, diag_max=25.0, offdiag_scale=0.01)
    rng1 = np.random.default_rng(100)
    P1 = rng1.normal(size=ndofs1)
    bc1 = {0: [True, True, True, True, True, True]}
    u1, r1 = fcn(P1.copy(), K1.copy(), bc1, n_nodes1)
    assert isinstance(u1, np.ndarray) and isinstance(r1, np.ndarray)
    assert u1.shape == (ndofs1,)
    assert r1.shape == (ndofs1,)
    fixed1, free1 = partition_indices(bc1, n_nodes1)
    Kff1 = K1[np.ix_(free1, free1)]
    Ksf1 = K1[np.ix_(fixed1, free1)]
    Pf1 = P1[free1]
    Ps1 = P1[fixed1]
    u_free_expected1 = np.linalg.solve(Kff1, Pf1)
    u_expected1 = np.zeros(ndofs1)
    u_expected1[free1] = u_free_expected1
    r_expected1 = np.zeros(ndofs1)
    r_expected1[fixed1] = Ksf1 @ u_free_expected1 - Ps1
    assert np.allclose(u1, u_expected1, rtol=1e-10, atol=1e-12)
    assert np.allclose(r1, r_expected1, rtol=1e-10, atol=1e-12)
    assert np.allclose(u1[fixed1], 0.0, atol=1e-12)
    assert np.allclose(r1[free1], 0.0, atol=1e-12)
    assert np.allclose(Kff1 @ u1[free1], Pf1, rtol=1e-10, atol=1e-12)
    assert np.allclose(K1 @ u1, P1 + r1, rtol=1e-10, atol=1e-12)
    n_nodes2 = 3
    ndofs2 = 6 * n_nodes2
    K2 = build_spd(ndofs2, seed=7, diag_min=12.0, diag_max=28.0, offdiag_scale=0.005)
    rng2 = np.random.default_rng(999)
    P2 = rng2.normal(size=ndofs2)
    bc2 = {0: [True, True, True, False, False, False], 1: [False, False, False, False, False, True]}
    u2, r2 = fcn(P2.copy(), K2.copy(), bc2, n_nodes2)
    assert isinstance(u2, np.ndarray) and isinstance(r2, np.ndarray)
    assert u2.shape == (ndofs2,)
    assert r2.shape == (ndofs2,)
    fixed2, free2 = partition_indices(bc2, n_nodes2)
    Kff2 = K2[np.ix_(free2, free2)]
    Ksf2 = K2[np.ix_(fixed2, free2)]
    Pf2 = P2[free2]
    Ps2 = P2[fixed2]
    u_free_expected2 = np.linalg.solve(Kff2, Pf2)
    u_expected2 = np.zeros(ndofs2)
    u_expected2[free2] = u_free_expected2
    r_expected2 = np.zeros(ndofs2)
    r_expected2[fixed2] = Ksf2 @ u_free_expected2 - Ps2
    assert np.allclose(u2, u_expected2, rtol=1e-10, atol=1e-12)
    assert np.allclose(r2, r_expected2, rtol=1e-10, atol=1e-12)
    assert np.allclose(u2[fixed2], 0.0, atol=1e-12)
    assert np.allclose(r2[free2], 0.0, atol=1e-12)
    assert np.allclose(Kff2 @ u2[free2], Pf2, rtol=1e-10, atol=1e-12)
    assert np.allclose(K2 @ u2, P2 + r2, rtol=1e-10, atol=1e-12)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    ndofs = 6 * n_nodes
    D_fixed = np.full(6, 5.0)
    D_free = np.array([1e-22, 1e-20, 1e-18, 1e-16, 1e-10, 1.0])
    K = np.diag(np.concatenate([D_fixed, D_free]))
    rng = np.random.default_rng(2025)
    P = rng.normal(size=ndofs)
    bc = {0: [True, True, True, True, True, True]}
    with pytest.raises(ValueError):
        fcn(P, K, bc, n_nodes)