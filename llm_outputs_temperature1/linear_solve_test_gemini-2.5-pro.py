def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
for small, solvable systems.
Verifies boundary conditions, internal equilibrium, and global force balance across
multiple cases with different free/fixed DOF configurations."""
    K1 = np.array([[100.0, -100.0], [-100.0, 100.0]])
    P1 = np.array([0.0, 50.0])
    fixed1 = [0]
    free1 = [1]
    u_expected1 = np.array([0.0, 0.5])
    R_expected1 = np.array([-50.0, 0.0])
    K2 = np.array([[4.0, 2.0], [2.0, 3.0]])
    P2 = np.array([1.0, 2.0])
    fixed2 = [0]
    free2 = [1]
    u_expected2 = np.array([0.0, 2.0 / 3.0])
    R_expected2 = np.array([1.0 / 3.0, 0.0])
    K3 = K2
    P3 = P2
    fixed3 = [1]
    free3 = [0]
    u_expected3 = np.array([0.25, 0.0])
    R_expected3 = np.array([0.0, -1.5])
    K4 = np.array([[10.0, -2.0, 0.0], [-2.0, 5.0, 1.0], [0.0, 1.0, 8.0]])
    P4 = np.array([5.0, 10.0, 15.0])
    fixed4 = [0, 2]
    free4 = [1]
    u_expected4 = np.array([0.0, 2.0, 0.0])
    R_expected4 = np.array([-9.0, 0.0, -13.0])
    K5 = np.array([[4.0, 1.0], [1.0, 3.0]])
    P5 = np.array([5.0, 4.0])
    fixed5 = []
    free5 = [0, 1]
    u_expected5 = np.linalg.solve(K5, P5)
    R_expected5 = np.array([0.0, 0.0])
    test_cases = [(P1, K1, fixed1, free1, u_expected1, R_expected1), (P2, K2, fixed2, free2, u_expected2, R_expected2), (P3, K3, fixed3, free3, u_expected3, R_expected3), (P4, K4, fixed4, free4, u_expected4, R_expected4), (P5, K5, fixed5, free5, u_expected5, R_expected5)]
    for (P, K, fixed, free, u_exp, R_exp) in test_cases:
        (u_actual, R_actual) = fcn(P, K, fixed, free)
        assert u_actual.shape == u_exp.shape
        assert R_actual.shape == R_exp.shape
        np.testing.assert_allclose(u_actual, u_exp, atol=1e-09)
        np.testing.assert_allclose(R_actual, R_exp, atol=1e-09)
        if fixed:
            np.testing.assert_allclose(u_actual[fixed], 0)
        if free:
            np.testing.assert_allclose(R_actual[free], 0)
        np.testing.assert_allclose(K @ u_actual, P + R_actual, atol=1e-09)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
(i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
to a numerically reliable degree.
This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
by using fixed/free DOF partitioning, and checks that the function does not proceed with
solving but instead raises the documented ValueError."""
    K_ill1 = np.array([[100.0, -100.0], [-100.0, 100.0]])
    P_ill1 = np.array([10.0, -10.0])
    fixed_ill1 = []
    free_ill1 = [0, 1]
    K_ill2 = np.array([[10.0, 2.0, 3.0], [2.0, 4.0, 6.0], [3.0, 6.0, 9.0]])
    P_ill2 = np.array([1.0, 2.0, 3.0])
    fixed_ill2 = [0]
    free_ill2 = [1, 2]
    epsilon = 1e-17
    K_ill3 = np.array([[10.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0 + epsilon]])
    P_ill3 = np.array([1.0, 1.0, 1.0])
    fixed_ill3 = [0]
    free_ill3 = [1, 2]
    test_cases = [(P_ill1, K_ill1, fixed_ill1, free_ill1), (P_ill2, K_ill2, fixed_ill2, free_ill2), (P_ill3, K_ill3, fixed_ill3, free_ill3)]
    for (P, K, fixed, free) in test_cases:
        with pytest.raises(ValueError, match='ill-conditioned'):
            fcn(P, K, fixed, free)