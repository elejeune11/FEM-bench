def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations."""
    K = np.array([[2.0, -1.0], [-1.0, 2.0]])
    P = np.array([10.0, 0.0])
    fixed = np.array([1])
    free = np.array([0])
    (u, R) = fcn(P, K, fixed, free)
    assert u[free] == pytest.approx(5.0)
    assert u[fixed] == pytest.approx(0.0)
    assert R[fixed] == pytest.approx(-5.0)
    assert np.abs(np.sum(P) + np.sum(R)) < 1e-10
    K = np.array([[3.0, -1.0, -1.0], [-1.0, 3.0, -1.0], [-1.0, -1.0, 3.0]])
    P = np.array([10.0, 0.0, 0.0])
    fixed = np.array([1, 2])
    free = np.array([0])
    (u, R) = fcn(P, K, fixed, free)
    assert u[free] == pytest.approx(10 / 3)
    assert all(u[fixed] == 0.0)
    assert np.allclose(K @ u, P + R)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError."""
    K = np.array([[1e-08, 1.0], [1.0, 100000000.0]])
    P = np.array([1.0, 1.0])
    fixed = np.array([0])
    free = np.array([1])
    with pytest.raises(ValueError):
        fcn(P, K, fixed, free)
    K = np.array([[1.0, 1.0], [1.0, 1.0]])
    P = np.array([1.0, 1.0])
    fixed = np.array([0])
    free = np.array([1])
    with pytest.raises(ValueError):
        fcn(P, K, fixed, free)