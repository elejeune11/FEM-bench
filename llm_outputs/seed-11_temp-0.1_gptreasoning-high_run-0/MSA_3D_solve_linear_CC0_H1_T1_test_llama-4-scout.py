def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    P_global = np.array([0, 0, 0, 0, 0, 0, 0, -1, 0, 0, 0, 0])
    K_global = np.eye(12)
    boundary_conditions = {0: [True, True, True, True, True, True], 1: [False, False, False, False, False, False]}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert np.allclose(u[:6], 0)
    free_dofs = [6, 7, 8, 9, 10, 11]
    P_f = P_global[free_dofs]
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    assert np.allclose(np.dot(K_ff, u[free_dofs]), P_f)
    R_global = np.dot(K_global, u) - P_global
    assert np.allclose(R_global[:6], r[:6])
    assert np.allclose(R_global[6:], r[6:])

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 1
    P_global = np.array([0, 0, 0, 0, 0, 0])
    K_global = np.eye(6) * 1e-16
    boundary_conditions = {0: [False, False, False, False, False, False]}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)