def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems."""
    K = np.array([[2.0, -1.0], [-1.0, 2.0]])
    P = np.array([10.0, 0.0])
    fixed = np.array([1])
    free = np.array([0])
    (u, R) = fcn(P, K, fixed, free)
    assert_allclose(u[free], 6.666666666666667, rtol=1e-10)
    assert_allclose(u[fixed], 0.0, rtol=1e-10)
    assert_allclose(R[fixed], -3.333333333333334, rtol=1e-10)
    assert_allclose(R[free], 0.0, rtol=1e-10)
    K = np.array([[3.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 3.0]])
    P = np.array([12.0, 0.0, 0.0])
    fixed = np.array([1, 2])
    free = np.array([0])
    (u, R) = fcn(P, K, fixed, free)
    assert_allclose(u[free], 4.0, rtol=1e-10)
    assert_allclose(u[fixed], 0.0, rtol=1e-10)
    assert_allclose(R[fixed], [-4.0, 0.0], rtol=1e-10)
    assert_allclose(R[free], 0.0, rtol=1e-10)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that linear_solve raises a ValueError when the submatrix K_ff is ill-conditioned."""
    K = np.array([[1e-08, 1.0], [1.0, 100000000.0]])
    P = np.array([1.0, 1.0])
    fixed = np.array([0])
    free = np.array([1])
    try:
        fcn(P, K, fixed, free)
        assert False, 'Should have raised ValueError'
    except ValueError:
        pass
    K = np.array([[1.0, 1.0, 1.0], [1.0, 1.0, 1.0], [1.0, 1.0, 1.0]])
    P = np.array([1.0, 1.0, 1.0])
    fixed = np.array([0])
    free = np.array([1, 2])
    try:
        fcn(P, K, fixed, free)
        assert False, 'Should have raised ValueError'
    except ValueError:
        pass