def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations."""
    P_global1 = np.array([10, 0])
    K_global1 = np.array([[2, -1], [-1, 1]])
    fixed1 = [1]
    free1 = [0]
    (u1, r1) = fcn(P_global1, K_global1, fixed1, free1)
    np.testing.assert_allclose(u1, np.array([5, 0]))
    np.testing.assert_allclose(r1, np.array([5, 0]))
    P_global2 = np.array([10, 0, 0])
    K_global2 = np.array([[2, -1, 0], [-1, 2, -1], [0, -1, 1]])
    fixed2 = [1, 2]
    free2 = [0]
    (u2, r2) = fcn(P_global2, K_global2, fixed2, free2)
    np.testing.assert_allclose(u2, np.array([10, 0, 0]))
    np.testing.assert_allclose(r2, np.array([0, 10, 0]))

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError."""
    P_global = np.array([1, 0])
    K_global = np.array([[1e-17, 0], [0, 1]])
    fixed = [1]
    free = [0]
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)
    P_global = np.array([1, 0, 0])
    K_global = np.array([[1, 0, 0], [0, 0, 0], [0, 0, 1]])
    fixed = [1]
    free = [0, 2]
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)