def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global = np.diag(np.ones(n_dofs))
    u_expected = np.array([0.1, 0.2, 0.3, 0.01, 0.02, 0.03, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    P_global = K_global @ u_expected
    boundary_conditions = {1: [True] * 6}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    np.testing.assert_allclose(u, u_expected, atol=1e-10)
    residual = K_global @ u - P_global
    np.testing.assert_allclose(residual, np.zeros(n_dofs), atol=1e-10)
    fixed_dof_indices = [6, 7, 8, 9, 10, 11]
    free_dof_indices = list(set(range(n_dofs)) - set(fixed_dof_indices))
    assert np.allclose(r[free_dof_indices], 0.0), 'Reactions should be zero at free DOFs'
    np.testing.assert_allclose(r[fixed_dof_indices], -P_global[fixed_dof_indices], atol=1e-10)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global = np.eye(n_dofs)
    free_dof_index = 0
    K_global[free_dof_index, free_dof_index] = 0.0
    P_global = np.random.rand(n_dofs)
    boundary_conditions = {1: [True] * 6}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)