def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_global = np.eye(n_dof) * 100.0
    for i in range(n_dof - 1):
        K_global[i, i + 1] = 10.0
        K_global[i + 1, i] = 10.0
    K_global = (K_global + K_global.T) / 2.0
    K_global += np.eye(n_dof) * 50.0
    P_global = np.zeros(n_dof)
    P_global[6] = 100.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert np.allclose(u[:6], 0.0), 'Fixed DOFs should have zero displacement'
    free_dofs = np.array([6, 7, 8, 9, 10, 11])
    fixed_dofs = np.array([0, 1, 2, 3, 4, 5])
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    K_fs = K_global[np.ix_(free_dofs, fixed_dofs)]
    P_f = P_global[free_dofs]
    u_f = u[free_dofs]
    u_s = u[fixed_dofs]
    equilibrium_check = K_ff @ u_f - (P_f - K_fs @ u_s)
    assert np.allclose(equilibrium_check, 0.0, atol=1e-10), 'Free DOFs should satisfy equilibrium equation'
    assert np.allclose(r[free_dofs], 0.0, atol=1e-10), 'Reactions should be zero at free DOFs'
    global_equilibrium = K_global @ u - P_global - r
    assert np.allclose(global_equilibrium, 0.0, atol=1e-10), 'Global equilibrium should be satisfied'

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 3
    n_dof = 6 * n_nodes
    K_global = np.eye(n_dof)
    for i in range(6, 12):
        K_global[i, i] = 1e-15
    P_global = np.ones(n_dof)
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(P_global, K_global, boundary_conditions, n_nodes)