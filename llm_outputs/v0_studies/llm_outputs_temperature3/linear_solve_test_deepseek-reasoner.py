def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations."""
    K1 = np.array([[2.0, -1.0], [-1.0, 2.0]])
    P1 = np.array([1.0, 0.0])
    fixed1 = [0]
    free1 = [1]
    (u1, R1) = fcn(P1, K1, fixed1, free1)
    assert u1[fixed1[0]] == 0.0
    residual1 = K1 @ u1 - (P1 + R1)
    assert np.allclose(residual1, 0.0, atol=1e-10)
    K2 = np.array([[3.0, -1.0, -1.0], [-1.0, 2.0, 0.0], [-1.0, 0.0, 2.0]])
    P2 = np.array([0.0, 1.0, 2.0])
    fixed2 = [0]
    free2 = [1, 2]
    (u2, R2) = fcn(P2, K2, fixed2, free2)
    assert u2[fixed2[0]] == 0.0
    assert not np.allclose(u2[free2], 0.0)
    residual2 = K2 @ u2 - (P2 + R2)
    assert np.allclose(residual2, 0.0, atol=1e-10)
    K3 = np.eye(4) * 2.0 + np.eye(4, k=1) * -0.5 + np.eye(4, k=-1) * -0.5
    P3 = np.array([0.5, 1.0, 0.5, 1.0])
    fixed3 = [1, 3]
    free3 = [0, 2]
    (u3, R3) = fcn(P3, K3, fixed3, free3)
    assert np.all(u3[fixed3] == 0.0)
    assert np.all(R3[free3] == 0.0)
    residual3 = K3 @ u3 - (P3 + R3)
    assert np.allclose(residual3, 0.0, atol=1e-10)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError."""
    K_singular = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    P_singular = np.array([1.0, 2.0, 3.0])
    fixed_singular = [0, 2]
    free_singular = [1]
    with pytest.raises(ValueError):
        fcn(P_singular, K_singular, fixed_singular, free_singular)
    K_ill_conditioned = np.array([[1.0, 0.0, 0.0], [0.0, 1e-16, 1e-16], [0.0, 1e-16, 1e-16]])
    P_ill_conditioned = np.array([1.0, 2.0, 3.0])
    fixed_ill_conditioned = [0]
    free_ill_conditioned = [1, 2]
    with pytest.raises(ValueError):
        fcn(P_ill_conditioned, K_ill_conditioned, fixed_ill_conditioned, free_ill_conditioned)
    K_dependent = np.array([[2.0, 1.0, 1.0], [1.0, 2.0, 2.0], [1.0, 2.0, 2.0]])
    P_dependent = np.array([1.0, 2.0, 3.0])
    fixed_dependent = [0]
    free_dependent = [1, 2]
    with pytest.raises(ValueError):
        fcn(P_dependent, K_dependent, fixed_dependent, free_dependent)