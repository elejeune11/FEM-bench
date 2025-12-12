def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_global = np.eye(n_dof)
    P_global = np.zeros(n_dof)
    P_global[6] = 1.0
    boundary_conditions = {0: [True] * 6}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    expected_u = np.zeros(n_dof)
    expected_u[6] = 1.0
    np.testing.assert_array_almost_equal(u, expected_u)
    expected_r = np.zeros(n_dof)
    expected_r[0] = -1.0
    np.testing.assert_array_almost_equal(r, expected_r)
    assert np.allclose(P_global + r, 0.0)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_global = np.zeros((n_dof, n_dof))
    P_global = np.ones(n_dof)
    boundary_conditions = {0: [True] * 6}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)