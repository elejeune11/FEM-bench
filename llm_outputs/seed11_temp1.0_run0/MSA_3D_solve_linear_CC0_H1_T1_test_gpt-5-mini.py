def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Verifies linear_solve against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    n_dof = 6 * n_nodes
    rng = np.random.RandomState(0)
    A = rng.randn(n_dof, n_dof)
    K_global = A.T @ A + np.eye(n_dof) * 0.001
    P_global = np.zeros(n_dof)
    P_global[6 * 1 + 0] = 5.0
    boundary_conditions = {0: np.ones(6, dtype=bool)}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert isinstance(u, np.ndarray) and u.shape == (n_dof,)
    assert isinstance(r, np.ndarray) and r.shape == (n_dof,)
    fixed_indices = []
    free_indices = []
    for node in range(n_nodes):
        fixed_mask = np.array(boundary_conditions.get(node, np.zeros(6, dtype=bool)), dtype=bool)
        for dof in range(6):
            idx = node * 6 + dof
            if fixed_mask[dof]:
                fixed_indices.append(idx)
            else:
                free_indices.append(idx)
    fixed_indices = np.array(fixed_indices, dtype=int)
    free_indices = np.array(free_indices, dtype=int)
    assert np.allclose(u[fixed_indices], 0.0)
    K_ff = K_global[np.ix_(free_indices, free_indices)]
    u_f = u[free_indices]
    P_f = P_global[free_indices]
    assert np.allclose(K_ff @ u_f, P_f, atol=1e-08, rtol=1e-06)
    if fixed_indices.size > 0:
        K_sf = K_global[np.ix_(fixed_indices, free_indices)]
        P_fixed = P_global[fixed_indices]
        r_expected_fixed = K_sf @ u_f - P_fixed
        assert np.allclose(r[fixed_indices], r_expected_fixed, atol=1e-08, rtol=1e-06)
        assert np.allclose(r[free_indices], 0.0)
    assert np.allclose(K_global @ u - P_global, r, atol=1e-08, rtol=1e-06)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned."""
    n_nodes = 1
    n_dof = 6 * n_nodes
    boundary_conditions = {}
    diag = np.ones(n_dof)
    diag[-1] = 1e-20
    K_global = np.diag(diag)
    P_global = np.zeros(n_dof)
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)