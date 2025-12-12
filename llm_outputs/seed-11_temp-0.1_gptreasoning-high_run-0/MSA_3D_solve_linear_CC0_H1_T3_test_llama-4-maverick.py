def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    K_global = np.array([[4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 2, 0, -4, 0, 0, 0, 2], [0, 0, 4, 0, -2, 0, 0, 0, -4, 0, -2, 0], [0, 0, 0, 4, 0, 0, 0, 0, 0, -4, 0, 0], [0, 0, -2, 0, 4, 0, 0, 0, 2, 0, -2, 0], [0, 2, 0, 0, 0, 4, 0, -2, 0, 0, 0, 2], [-4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0], [0, -4, 0, 0, 0, -2, 0, 4, 0, 0, 0, -2], [0, 0, -4, 0, 2, 0, 0, 0, 4, 0, 2, 0], [0, 0, 0, -4, 0, 0, 0, 0, 0, 4, 0, 0], [0, 0, -2, 0, -2, 0, 0, 0, 2, 0, 4, 0], [0, 2, 0, 0, 0, 2, 0, -2, 0, 0, 0, 4]])
    P_global = np.array([0, 0, 0, 0, 0, 0, 1, 2, 3, 4, 5, 6])
    boundary_conditions = {0: np.ones(6, dtype=bool)}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert np.allclose(u[:6], 0)
    assert np.allclose(r[:6], np.array([-1, -2, -3, -4, -5, -6]))
    assert np.allclose(np.dot(K_global, u), r + P_global)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    K_global = np.array([[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]])
    P_global = np.ones(12)
    boundary_conditions = {0: np.array([True] * 3 + [False] * 3)}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)