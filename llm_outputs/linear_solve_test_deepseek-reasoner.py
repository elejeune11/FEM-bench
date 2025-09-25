def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    K1 = np.array([[4, -2], [-2, 4]], dtype=float)
    P1 = np.array([6, 0], dtype=float)
    fixed1 = [1]
    free1 = [0]
    (u1, r1) = fcn(P1, K1, fixed1, free1)
    expected_u1 = np.array([1.5, 0.0])
    expected_r1 = np.array([0.0, 3.0])
    np.testing.assert_allclose(u1, expected_u1, rtol=1e-10)
    np.testing.assert_allclose(r1, expected_r1, rtol=1e-10)
    np.testing.assert_allclose(P1, K1 @ u1, rtol=1e-10)
    K2 = np.array([[3, -1, -1], [-1, 2, 0], [-1, 0, 2]], dtype=float)
    P2 = np.array([2, 0, 0], dtype=float)
    fixed2 = [0]
    free2 = [1, 2]
    (u2, r2) = fcn(P2, K2, fixed2, free2)
    assert u2[fixed2[0]] == 0.0
    np.testing.assert_allclose(P2, K2 @ u2, rtol=1e-10)
    K3 = np.eye(4, dtype=float) * 2 + np.eye(4, k=1, dtype=float) * 0.5 + np.eye(4, k=-1, dtype=float) * 0.5
    P3 = np.array([1, 2, 3, 4], dtype=float)
    fixed3 = [0, 2]
    free3 = [1, 3]
    (u3, r3) = fcn(P3, K3, fixed3, free3)
    assert u3[fixed3[0]] == 0.0
    assert u3[fixed3[1]] == 0.0
    np.testing.assert_allclose(P3, K3 @ u3, rtol=1e-10)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError.
    """
    K_singular = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    P = np.array([1, 2, 3], dtype=float)
    fixed = [2]
    free = [0, 1]
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(P, K_singular, fixed, free)
    K_nearly_singular = np.array([[1, 1], [1, 1 + 1e-16]], dtype=float)
    P2 = np.array([1, 1], dtype=float)
    fixed2 = []
    free2 = [0, 1]
    with pytest.raises(ValueError):
        fcn(P2, K_nearly_singular, fixed2, free2)
    n = 5
    K_ill = np.ones((n, n), dtype=float) * 1e-15 + np.eye(n) * 1e-16
    P3 = np.ones(n, dtype=float)
    fixed3 = []
    free3 = list(range(n))
    with pytest.raises(ValueError):
        fcn(P3, K_ill, fixed3, free3)