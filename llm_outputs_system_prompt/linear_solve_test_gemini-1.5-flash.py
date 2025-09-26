def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    P_global1 = np.array([10, 0, 0, 0])
    K_global1 = np.array([[2, -1, 0, 0], [-1, 2, -1, 0], [0, -1, 2, -1], [0, 0, -1, 1]])
    fixed1 = [0]
    free1 = [1, 2, 3]
    (u1, r1) = fcn(P_global1, K_global1, fixed1, free1)
    np.testing.assert_allclose(u1[fixed1], 0)
    np.testing.assert_allclose(r1[free1], 0)
    np.testing.assert_allclose(np.sum(r1), 10)
    P_global2 = np.array([1, 2, 3, 4, 5, 6])
    K_global2 = np.array([[10, -1, 0, 0, 0, 0], [-1, 10, -2, 0, 0, 0], [0, -2, 10, -3, 0, 0], [0, 0, -3, 10, -4, 0], [0, 0, 0, -4, 10, -5], [0, 0, 0, 0, -5, 10]])
    fixed2 = [0, 5]
    free2 = [1, 2, 3, 4]
    (u2, r2) = fcn(P_global2, K_global2, fixed2, free2)
    np.testing.assert_allclose(u2[fixed2], 0)
    np.testing.assert_allclose(r2[free2], 0)
    np.testing.assert_allclose(np.sum(r2), np.sum(P_global2))

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError.
    """
    P_global = np.array([1, 2, 3, 4])
    K_global = np.array([[1, 0, 0, 0], [0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    fixed = [1]
    free = [0, 2, 3]
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)
    K_global_near_singular = np.array([[1e-17, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    with pytest.raises(ValueError):
        fcn(P_global, K_global_near_singular, fixed, free)