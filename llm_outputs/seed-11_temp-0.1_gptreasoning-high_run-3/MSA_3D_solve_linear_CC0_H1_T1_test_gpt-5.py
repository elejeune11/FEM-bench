def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """

    def build_fixed_free_indices(boundary_conditions, n_nodes):
        fixed = []
        for node, bc in boundary_conditions.items():
            for i, b in enumerate(bc):
                if b:
                    fixed.append(node * 6 + i)
        all_idx = np.arange(6 * n_nodes, dtype=int)
        fixed = np.array(sorted(fixed), dtype=int)
        free = np.array(sorted(list(set(all_idx) - set(fixed))), dtype=int)
        return (fixed, free)
    rng = np.random.default_rng(12345)
    n_nodes = 2
    dof = 6 * n_nodes
    A = rng.standard_normal((dof, dof))
    K = A.T @ A + 10.0 * np.eye(dof)
    P = rng.standard_normal(dof)
    boundary_conditions = {0: [True, True, True, True, True, True]}
    u, r = fcn(P, K, boundary_conditions, n_nodes)
    fixed, free = build_fixed_free_indices(boundary_conditions, n_nodes)
    assert u.shape == (dof,)
    assert r.shape == (dof,)
    assert np.allclose(u[fixed], 0.0)
    K_ff = K[np.ix_(free, free)]
    K_sf = K[np.ix_(fixed, free)]
    P_f = P[free]
    P_s = P[fixed]
    u_f = u[free]
    r_s_expected = K_sf @ u_f - P_s
    assert np.allclose(K_ff @ u_f, P_f, rtol=1e-08, atol=1e-10)
    assert np.allclose(r[fixed], r_s_expected, rtol=1e-08, atol=1e-10)
    assert np.allclose(r[free], 0.0)
    assert np.allclose(K @ u, P + r, rtol=1e-08, atol=1e-10)
    n_nodes = 3
    dof = 6 * n_nodes
    A = rng.standard_normal((dof, dof))
    K = A.T @ A + 10.0 * np.eye(dof)
    P = rng.standard_normal(dof)
    boundary_conditions = {0: [True, True, True, True, True, True], 1: [True, False, False, False, True, False]}
    u, r = fcn(P, K, boundary_conditions, n_nodes)
    fixed, free = build_fixed_free_indices(boundary_conditions, n_nodes)
    assert u.shape == (dof,)
    assert r.shape == (dof,)
    assert np.allclose(u[fixed], 0.0)
    K_ff = K[np.ix_(free, free)]
    K_sf = K[np.ix_(fixed, free)]
    P_f = P[free]
    P_s = P[fixed]
    u_f = u[free]
    r_s_expected = K_sf @ u_f - P_s
    assert np.allclose(K_ff @ u_f, P_f, rtol=1e-08, atol=1e-10)
    assert np.allclose(r[fixed], r_s_expected, rtol=1e-08, atol=1e-10)
    assert np.allclose(r[free], 0.0)
    assert np.allclose(K @ u, P + r, rtol=1e-08, atol=1e-10)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    dof = 6 * n_nodes
    K = np.zeros((dof, dof))
    P = np.zeros(dof)
    boundary_conditions = {0: [True, True, True, True, True, True]}
    with pytest.raises(ValueError):
        fcn(P, K, boundary_conditions, n_nodes)