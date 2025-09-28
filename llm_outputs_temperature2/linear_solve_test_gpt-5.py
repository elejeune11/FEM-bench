def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium (free DOFs satisfy K u = P),
    and that the returned reaction vector matches K u - P at fixed DOFs across
    multiple cases with different free/fixed DOF configurations.
    """
    K1 = np.array([[4.0, 1.0, 0.0], [1.0, 3.0, 1.0], [0.0, 1.0, 2.0]])
    P1 = np.array([1.0, 2.0, 3.0])
    fixed1 = np.array([0])
    free1 = np.array([1, 2])
    K2 = np.array([[6.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 4.0]])
    P2 = np.array([0.5, -1.0, 2.0])
    fixed2 = np.array([0, 2])
    free2 = np.array([1])
    B3 = np.array([[2.0, 1.0, 0.0, 1.0], [0.0, 1.0, 1.0, 0.0], [1.0, 0.0, 2.0, 1.0], [0.0, 1.0, 0.0, 2.0]])
    K3 = B3.T @ B3 + 3.0 * np.eye(4)
    P3 = np.array([2.0, -3.0, 1.0, 4.0])
    fixed3 = np.array([1, 3])
    free3 = np.array([0, 2])
    for (K, P, fixed, free) in [(K1, P1, fixed1, free1), (K2, P2, fixed2, free2), (K3, P3, fixed3, free3)]:
        K_ff = K[np.ix_(free, free)]
        P_f = P[free]
        assert np.linalg.cond(K_ff) < 1e+16
        u_free_expected = np.linalg.solve(K_ff, P_f)
        u_expected = np.zeros_like(P)
        u_expected[free] = u_free_expected
        (u, R) = fcn(P, K, fixed, free)
        assert u.shape == P.shape
        assert R.shape == P.shape
        assert np.allclose(u[fixed], 0.0)
        assert np.allclose(u[free], u_free_expected)
        residual = K @ u - P
        assert np.allclose(residual[free], 0.0)
        R_expected = residual
        assert np.allclose(R, R_expected)
        assert np.allclose(R[free], 0.0)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """
    Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (condition number >= 1e16), indicating that the linear system cannot be solved reliably.
    Constructs a case where `K_ff` is singular by partitioning the DOFs such that the free-free
    block is rank-deficient.
    """
    K = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    P = np.array([1.0, 2.0, 3.0])
    fixed = np.array([2])
    free = np.array([0, 1])
    K_ff = K[np.ix_(free, free)]
    assert not np.isfinite(np.linalg.cond(K_ff)) or np.linalg.cond(K_ff) >= 1e+16
    with pytest.raises(ValueError):
        fcn(P, K, fixed, free)