def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `MSA_3D_solve_linear_CC0_H1_T3` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    P_global = np.array([0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0])
    K_global = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]])
    boundary_conditions = {0: [True, True, True, True, True, True], 1: [False, False, False, False, False, False]}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert np.allclose(K_global @ u, P_global)
    free_dofs = [False, False, False, False, False, False, True, True, True, True, True, True]
    assert np.allclose(u[free_dofs], np.linalg.solve(K_global[np.ix_(free_dofs, free_dofs)], P_global[free_dofs]))
    fixed_dofs = [True, True, True, True, True, True, False, False, False, False, False, False]
    assert np.allclose(r, K_global[np.ix_(fixed_dofs, ~fixed_dofs)] @ u[~fixed_dofs] - P_global[fixed_dofs])

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 1
    P_global = np.zeros(6)
    K_global = np.eye(6) * 1e-10
    boundary_conditions = {0: [False, False, False, False, False, False]}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)