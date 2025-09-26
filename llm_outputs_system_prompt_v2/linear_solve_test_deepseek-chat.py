def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations."""
    K1 = np.array([[2.0, -1.0], [-1.0, 2.0]])
    P1 = np.array([1.0, 0.0])
    fixed1 = [0]
    free1 = [1]
    (u1, r1) = fcn(P1, K1, fixed1, free1)
    expected_u1 = np.array([0.0, 0.5])
    expected_r1 = np.array([-0.5, 0.0])
    np.testing.assert_allclose(u1, expected_u1, rtol=1e-10)
    np.testing.assert_allclose(r1, expected_r1, rtol=1e-10)
    residual = P1 - (K1 @ u1 + r1)
    np.testing.assert_allclose(residual, np.zeros_like(P1), atol=1e-10)
    K2 = np.array([[3.0, -1.0, -1.0], [-1.0, 2.0, 0.0], [-1.0, 0.0, 2.0]])
    P2 = np.array([0.0, 1.0, 2.0])
    fixed2 = [0]
    free2 = [1, 2]
    (u2, r2) = fcn(P2, K2, fixed2, free2)
    assert u2[0] == 0.0
    assert r2[1] == 0.0
    assert r2[2] == 0.0
    residual2 = P2 - (K2 @ u2 + r2)
    np.testing.assert_allclose(residual2, np.zeros_like(P2), atol=1e-10)
    K3 = np.array([[4.0, -1.0, 0.0, 0.0], [-1.0, 4.0, -1.0, 0.0], [0.0, -1.0, 4.0, -1.0], [0.0, 0.0, -1.0, 4.0]])
    P3 = np.array([1.0, 0.0, 0.0, 1.0])
    fixed3 = [0, 2]
    free3 = [1, 3]
    (u3, r3) = fcn(P3, K3, fixed3, free3)
    assert u3[0] == 0.0
    assert u3[2] == 0.0
    assert r3[1] == 0.0
    assert r3[3] == 0.0
    residual3 = P3 - (K3 @ u3 + r3)
    np.testing.assert_allclose(residual3, np.zeros_like(P3), atol=1e-10)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError."""
    K_singular = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
    P_singular = np.array([1.0, 1.0, 1.0])
    fixed_singular = [0]
    free_singular = [1, 2]
    with pytest.raises(ValueError):
        fcn(P_singular, K_singular, fixed_singular, free_singular)
    K_nearly_singular = np.array([[1e-16, 0.0, 0.0], [0.0, 1e-16, 0.0], [0.0, 0.0, 1.0]])
    P_nearly_singular = np.array([1.0, 1.0, 1.0])
    fixed_nearly_singular = [0]
    free_nearly_singular = [1, 2]
    with pytest.raises(ValueError):
        fcn(P_nearly_singular, K_nearly_singular, fixed_nearly_singular, free_nearly_singular)
    K_zero_block = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    P_zero_block = np.array([1.0, 1.0, 1.0])
    fixed_zero_block = [0]
    free_zero_block = [1, 2]
    with pytest.raises(ValueError):
        fcn(P_zero_block, K_zero_block, fixed_zero_block, free_zero_block)