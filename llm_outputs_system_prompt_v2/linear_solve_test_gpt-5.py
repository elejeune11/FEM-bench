def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    K = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]])
    P = np.array([1.0, 2.0, 3.0])
    cases = [(np.array([0], dtype=int), np.array([1, 2], dtype=int)), (np.array([1], dtype=int), np.array([0, 2], dtype=int)), (np.array([0, 2], dtype=int), np.array([1], dtype=int))]
    for (fixed, free) in cases:
        (u, R) = fcn(P.copy(), K.copy(), fixed.copy(), free.copy())
        assert u.shape == P.shape
        assert R.shape == P.shape
        assert np.allclose(u[fixed], 0.0)
        K_ff = K[np.ix_(free, free)]
        P_f = P[free]
        u_f_expected = np.linalg.solve(K_ff, P_f)
        assert np.allclose(u[free], u_f_expected)
        assert np.allclose(R[free], 0.0)
        K_if = K[np.ix_(fixed, free)]
        P_i = P[fixed]
        R_i_expected = K_if @ u_f_expected - P_i
        assert np.allclose(R[fixed], R_i_expected)
        residual = K @ u - P
        assert np.allclose(residual, R)
        assert np.linalg.norm(residual - R) <= 1e-12

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """
    Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError.
    """
    K = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 2.0]])
    P = np.array([0.5, 1.0, -0.5])
    fixed = np.array([0, 2], dtype=int)
    free = np.array([1], dtype=int)
    with pytest.raises(ValueError):
        fcn(P.copy(), K.copy(), fixed.copy(), free.copy())
    fixed_singular = np.array([2], dtype=int)
    free_singular = np.array([0, 1], dtype=int)
    K_ff = K[np.ix_(free_singular, free_singular)]
    assert not np.isfinite(np.linalg.cond(K_ff))
    with pytest.raises(ValueError):
        fcn(P.copy(), K.copy(), fixed_singular.copy(), free_singular.copy())