def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    K1 = np.array([[2, -1], [-1, 2]], dtype=float)
    P1 = np.array([1.0, 0.0])
    fixed1 = [0]
    free1 = [1]
    (u1, reactions1) = fcn(P1, K1, fixed1, free1)
    assert u1[fixed1[0]] == 0.0
    assert np.allclose(K1 @ u1, P1 + reactions1, atol=1e-10)
    assert np.allclose(reactions1[free1], 0.0, atol=1e-10)
    K2 = np.array([[3, -1, -1], [-1, 2, 0], [-1, 0, 2]], dtype=float)
    P2 = np.array([0.0, 2.0, 1.0])
    fixed2 = [0, 2]
    free2 = [1]
    (u2, reactions2) = fcn(P2, K2, fixed2, free2)
    assert np.allclose(u2[fixed2], 0.0, atol=1e-10)
    assert np.allclose(K2 @ u2, P2 + reactions2, atol=1e-10)
    assert np.allclose(reactions2[free2], 0.0, atol=1e-10)
    K3 = np.array([[4, -2], [-2, 4]], dtype=float)
    P3 = np.array([3.0, 1.0])
    fixed3 = []
    free3 = [0, 1]
    (u3, reactions3) = fcn(P3, K3, fixed3, free3)
    assert np.allclose(reactions3, 0.0, atol=1e-10)
    assert np.allclose(K3 @ u3, P3, atol=1e-10)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError.
    """
    K_singular = np.array([[1, 1], [1, 1]], dtype=float)
    P = np.array([1.0, 1.0])
    fixed = [0]
    free = [1]
    (u, reactions) = fcn(P, K_singular, fixed, free)
    K_ill_conditioned = np.array([[1, 1e+16], [1e+16, 1e+32]], dtype=float)
    P2 = np.array([1.0, 2.0])
    fixed2 = []
    free2 = [0, 1]
    with pytest.raises(ValueError):
        fcn(P2, K_ill_conditioned, fixed2, free2)
    K_nearly_singular = np.array([[1, 1], [1, 1 + 1e-15]], dtype=float)
    P3 = np.array([1.0, 1.0])
    fixed3 = []
    free3 = [0, 1]
    with pytest.raises(ValueError):
        fcn(P3, K_nearly_singular, fixed3, free3)