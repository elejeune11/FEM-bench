def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `MSA_3D_solve_linear_CC0_H1_T1` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    K_global = np.array([[4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0], [0, 4, 0, 0, 0, 0, 0, -4, 0, 0, 0, 0], [0, 0, 4, 0, 0, 0, 0, 0, -4, 0, 0, 0], [0, 0, 0, 4, 0, 0, 0, 0, 0, -4, 0, 0], [0, 0, 0, 0, 4, 0, 0, 0, 0, 0, -4, 0], [0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0, -4], [-4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0, 0], [0, -4, 0, 0, 0, 0, 0, 4, 0, 0, 0, 0], [0, 0, -4, 0, 0, 0, 0, 0, 4, 0, 0, 0], [0, 0, 0, -4, 0, 0, 0, 0, 0, 4, 0, 0], [0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 4, 0], [0, 0, 0, 0, 0, -4, 0, 0, 0, 0, 0, 4]], dtype=float)
    P_global = np.array([0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0], dtype=float)
    boundary_conditions = {0: np.ones(6, dtype=bool)}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert np.allclose(u[:6], 0)
    assert not np.allclose(u[6:], 0)
    K_ff = K_global[6:, 6:]
    u_f = u[6:]
    P_f = P_global[6:]
    assert np.allclose(np.dot(K_ff, u_f), P_f)
    r_expected = np.dot(K_global[:6, 6:], u_f) - P_global[:6]
    assert np.allclose(r[:6], r_expected)
    assert np.allclose(r[6:], 0)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    K_global = np.array([[1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1], [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0], [0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0], [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1]], dtype=float)
    P_global = np.ones(12, dtype=float)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)