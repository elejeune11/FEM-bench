def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    n_dofs = n_nodes * 6
    np.random.seed(123)
    A = np.random.rand(n_dofs, n_dofs)
    K_global = A.T @ A + 0.1 * np.eye(n_dofs)
    P_global = np.random.rand(n_dofs)
    boundary_conditions = {0: np.array([True, True, True, True, True, True], dtype=bool)}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    fixed_dof_indices = np.arange(6)
    free_dof_indices = np.arange(6, 12)
    assert np.allclose(u[fixed_dof_indices], 0.0, atol=1e-12), 'Displacements at fixed supports must be zero.'
    K_ff = K_global[np.ix_(free_dof_indices, free_dof_indices)]
    P_f = P_global[free_dof_indices]
    u_f_computed = u[free_dof_indices]
    u_f_expected = np.linalg.solve(K_ff, P_f)
    assert np.allclose(u_f_computed, u_f_expected, atol=1e-10), 'Free displacements do not match the solution of K_ff * u_f = P_f.'
    assert np.allclose(r[free_dof_indices], 0.0, atol=1e-12), 'Reactions at free nodes must be zero.'
    K_sf = K_global[np.ix_(fixed_dof_indices, free_dof_indices)]
    P_fixed = P_global[fixed_dof_indices]
    r_fixed_expected = K_sf @ u_f_computed - P_fixed
    assert np.allclose(r[fixed_dof_indices], r_fixed_expected, atol=1e-10), 'Reaction forces at fixed supports are incorrect.'
    global_force_mismatch = K_global @ u - (P_global + r)
    assert np.allclose(global_force_mismatch, 0.0, atol=1e-10), 'Global equilibrium K*u = P+r is not satisfied.'

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 1
    n_dofs = 6
    K_global = np.zeros((n_dofs, n_dofs))
    P_global = np.ones(n_dofs)
    boundary_conditions = {}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(P_global, K_global, boundary_conditions, n_nodes)