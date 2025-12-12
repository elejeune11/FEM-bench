def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    test_cases = [(2, {0: [True] * 6}), (3, {0: [True] * 6, 2: [True, True, True, False, False, False]}), (1, {}), (2, {1: [True, False, True, False, True, False]})]
    for (n_nodes, boundary_conditions) in test_cases:
        np.random.seed(42)
        n_dof = 6 * n_nodes
        A = np.random.rand(n_dof, n_dof)
        K_global = A @ A.T + np.eye(n_dof)
        P_global = np.random.rand(n_dof) * 100
        all_dofs = set(range(n_dof))
        fixed_dofs_set = set()
        for (node_idx, constraints) in boundary_conditions.items():
            for (i, is_fixed) in enumerate(constraints):
                if is_fixed:
                    dof = 6 * node_idx + i
                    fixed_dofs_set.add(dof)
        fixed_dofs = np.array(sorted(list(fixed_dofs_set)), dtype=int)
        free_dofs = np.array(sorted(list(all_dofs - fixed_dofs_set)), dtype=int)
        P_global_mod = P_global.copy()
        if fixed_dofs.size > 0:
            P_global_mod[fixed_dofs] = 0.0
        (u, r) = fcn(P_global_mod, K_global, boundary_conditions, n_nodes)
        assert u.shape == (n_dof,)
        assert r.shape == (n_dof,)
        if fixed_dofs.size > 0:
            assert np.allclose(u[fixed_dofs], 0.0)
        if free_dofs.size > 0:
            assert np.allclose(r[free_dofs], 0.0)
        if free_dofs.size > 0:
            K_ff = K_global[np.ix_(free_dofs, free_dofs)]
            P_f = P_global_mod[free_dofs]
            u_f = u[free_dofs]
            assert np.allclose(K_ff @ u_f, P_f)
        if fixed_dofs.size > 0 and free_dofs.size > 0:
            K_sf = K_global[np.ix_(fixed_dofs, free_dofs)]
            P_s = P_global_mod[fixed_dofs]
            r_s = r[fixed_dofs]
            u_f = u[free_dofs]
            assert np.allclose(r_s, K_sf @ u_f - P_s)
        assert np.allclose(K_global @ u, P_global_mod + r)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    np.random.seed(123)
    n_nodes_1 = 2
    n_dof_1 = 6 * n_nodes_1
    P_global_1 = np.ones(n_dof_1)
    boundary_conditions_1 = {}
    K_singular = np.random.rand(n_dof_1, n_dof_1)
    K_singular[:, 1] = K_singular[:, 0] * 0.5
    K_singular = K_singular @ K_singular.T
    with np.errstate(divide='ignore', invalid='ignore'):
        assert np.linalg.cond(K_singular) > 1e+16
    with pytest.raises(ValueError, match='(?i)ill-conditioned|singular'):
        fcn(P_global_1, K_singular, boundary_conditions_1, n_nodes_1)
    n_nodes_2 = 3
    n_dof_2 = 6 * n_nodes_2
    P_global_2 = np.ones(n_dof_2)
    boundary_conditions_2 = {0: [True] * 6}
    K_global_2 = np.eye(n_dof_2)
    free_dofs_size = 12
    K_ff_singular = np.random.rand(free_dofs_size, free_dofs_size)
    K_ff_singular[:, 0] = 0
    K_ff_singular = K_ff_singular @ K_ff_singular.T
    free_dofs_indices = np.arange(6, 18)
    K_global_2[np.ix_(free_dofs_indices, free_dofs_indices)] = K_ff_singular
    K_ff_extracted = K_global_2[np.ix_(free_dofs_indices, free_dofs_indices)]
    with np.errstate(divide='ignore', invalid='ignore'):
        assert np.linalg.cond(K_ff_extracted) > 1e+16
    with pytest.raises(ValueError, match='(?i)ill-conditioned|singular'):
        fcn(P_global_2, K_global_2, boundary_conditions_2, n_nodes_2)