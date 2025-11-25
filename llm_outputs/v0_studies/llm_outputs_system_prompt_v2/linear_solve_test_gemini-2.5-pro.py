def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations."""
    (k1, k2, F) = (10.0, 20.0, 5.0)
    K_global_1 = np.array([[k1, -k1], [-k1, k1 + k2]])
    P_global_1 = np.array([0.0, F])
    fixed_1 = np.array([0])
    free_1 = np.array([1])
    u_f_1 = F / (k1 + k2)
    u_expected_1 = np.array([0.0, u_f_1])
    K_pf_1 = K_global_1[np.ix_(fixed_1, free_1)]
    P_p_1 = P_global_1[fixed_1]
    R_p_1 = K_pf_1 @ u_expected_1[free_1] - P_p_1
    r_expected_1 = np.zeros_like(P_global_1)
    r_expected_1[fixed_1] = R_p_1
    (k1, k2, F1, F2) = (1.0, 2.0, 10.0, 20.0)
    K_global_2 = np.array([[k1, -k1, 0], [-k1, k1 + k2, -k2], [0, -k2, k2]])
    P_global_2 = np.array([F1, 0.0, F2])
    fixed_2 = np.array([1])
    free_2 = np.array([0, 2])
    K_ff_2 = K_global_2[np.ix_(free_2, free_2)]
    P_f_2 = P_global_2[free_2]
    u_f_2 = np.linalg.solve(K_ff_2, P_f_2)
    u_expected_2 = np.zeros(3)
    u_expected_2[free_2] = u_f_2
    K_pf_2 = K_global_2[np.ix_(fixed_2, free_2)]
    P_p_2 = P_global_2[fixed_2]
    R_p_2 = K_pf_2 @ u_f_2 - P_p_2
    r_expected_2 = np.zeros_like(P_global_2)
    r_expected_2[fixed_2] = R_p_2
    cases = [(P_global_1, K_global_1, fixed_1, free_1, u_expected_1, r_expected_1), (P_global_2, K_global_2, fixed_2, free_2, u_expected_2, r_expected_2)]
    for (P_global, K_global, fixed, free, u_expected, r_expected) in cases:
        (u_actual, r_actual) = fcn(P_global, K_global, fixed, free)
        assert np.allclose(u_actual, u_expected)
        assert np.allclose(r_actual, r_expected)
        assert np.allclose(u_actual[fixed], 0.0)
        assert np.allclose(r_actual[free], 0.0)
        assert np.allclose(K_global @ u_actual, P_global + r_actual)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError."""
    k = 10.0
    K_singular = np.array([[k, -k], [-k, k]])
    P_singular = np.array([1.0, -1.0])
    fixed_singular = []
    free_singular = [0, 1]
    K_near_singular = np.array([[10000000000.0, 0, 0, 0], [0, 1.0, 1.0, 0], [0, 1.0, 1.0 + 1e-17, 0], [0, 0, 0, 10000000000.0]])
    P_near_singular = np.array([0.0, 1.0, 1.0, 0.0])
    fixed_near_singular = [0, 3]
    free_near_singular = [1, 2]
    cases = [(P_singular, K_singular, fixed_singular, free_singular), (P_near_singular, K_near_singular, fixed_near_singular, free_near_singular)]
    for (P_global, K_global, fixed, free) in cases:
        with pytest.raises(ValueError) as excinfo:
            fcn(P_global, K_global, fixed, free)
        assert 'ill-conditioned' in str(excinfo.value).lower() or 'singular' in str(excinfo.value).lower()