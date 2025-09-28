def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    K = np.array([[10.0, 2.0, 1.0, 0.5], [2.0, 9.0, 0.3, 0.8], [1.0, 0.3, 8.0, 1.5], [0.5, 0.8, 1.5, 7.0]])
    P = np.array([1.0, 2.0, -1.0, 0.5])
    cases = [(np.array([0, 1]), np.array([2, 3])), (np.array([1, 3]), np.array([0, 2])), (np.array([0]), np.array([1, 2, 3])), (np.array([2, 3]), np.array([0, 1]))]
    for (fixed, free) in cases:
        (u, R) = fcn(P, K, fixed, free)
        assert u.shape == P.shape
        assert R.shape == P.shape
        K_ff = K[np.ix_(free, free)]
        K_fc = K[np.ix_(free, fixed)]
        K_cf = K[np.ix_(fixed, free)]
        u_f_expected = np.linalg.solve(K_ff, P[free])
        assert np.allclose(u[free], u_f_expected, atol=1e-12, rtol=1e-12)
        assert np.allclose(u[fixed], 0.0, atol=1e-12, rtol=1e-12)
        R_c_expected = K_cf @ u_f_expected - P[fixed]
        assert np.allclose(R[fixed], R_c_expected, atol=1e-12, rtol=1e-12)
        assert np.allclose(R[free], 0.0, atol=1e-12, rtol=1e-12)
        residual = K @ u - P + R
        assert np.allclose(residual, 0.0, atol=1e-12, rtol=1e-12)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """
    Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular `K_ff` matrix by using fixed/free DOF partitioning,
    and checks that the function raises the documented ValueError.
    """
    K = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 3.0]])
    P = np.array([1.0, -1.0, 0.5])
    fixed = np.array([2])
    free = np.array([0, 1])
    with pytest.raises(ValueError):
        fcn(P, K, fixed, free)