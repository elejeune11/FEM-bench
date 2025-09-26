def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations."""
    K1 = np.array([[2.0, -1.0], [-1.0, 2.0]])
    P1 = np.array([0.0, 10.0])
    fixed1 = [0]
    free1 = [1]
    (u1, r1) = fcn(P1, K1, fixed1, free1)
    assert np.allclose(u1[fixed1], 0.0)
    assert np.allclose(u1[free1], 5.0)
    assert np.allclose(r1[fixed1], -5.0)
    assert np.allclose(r1[free1], 0.0)
    assert np.allclose(P1 + r1, K1 @ u1)
    K2 = np.array([[4.0, -2.0, -1.0], [-2.0, 5.0, -2.0], [-1.0, -2.0, 4.0]])
    P2 = np.array([0.0, 15.0, 0.0])
    fixed2 = [0, 2]
    free2 = [1]
    (u2, r2) = fcn(P2, K2, fixed2, free2)
    assert np.allclose(u2[fixed2], 0.0)
    assert np.allclose(u2[free2], 3.0)
    assert np.allclose(r2[free2], 0.0)
    assert np.allclose(P2 + r2, K2 @ u2)
    assert np.allclose(np.sum(P2 + r2), 0.0)
    K3 = np.array([[10.0, -3.0, -2.0, 0.0], [-3.0, 8.0, -2.0, -1.0], [-2.0, -2.0, 9.0, -3.0], [0.0, -1.0, -3.0, 7.0]])
    P3 = np.array([0.0, 20.0, 10.0, 0.0])
    fixed3 = [0, 3]
    free3 = [1, 2]
    (u3, r3) = fcn(P3, K3, fixed3, free3)
    assert np.allclose(u3[fixed3], 0.0)
    assert np.allclose(r3[free3], 0.0)
    assert np.allclose(P3 + r3, K3 @ u3)
    K_ff = K3[np.ix_(free3, free3)]
    P_f = P3[free3]
    u_f_expected = np.linalg.solve(K_ff, P_f)
    assert np.allclose(u3[free3], u_f_expected)
    K4 = np.array([[5.0, -2.0], [-2.0, 5.0]])
    P4 = np.array([10.0, 5.0])
    fixed4 = []
    free4 = [0, 1]
    (u4, r4) = fcn(P4, K4, fixed4, free4)
    assert np.allclose(r4, 0.0)
    u4_expected = np.linalg.solve(K4, P4)
    assert np.allclose(u4, u4_expected)
    assert np.allclose(P4, K4 @ u4)
    K5 = np.array([[6.0, -2.0, -1.0, 0.0, -1.0], [-2.0, 8.0, -3.0, -1.0, 0.0], [-1.0, -3.0, 10.0, -2.0, -2.0], [0.0, -1.0, -2.0, 7.0, -3.0], [-1.0, 0.0, -2.0, -3.0, 9.0]])
    P5 = np.array([0.0, 25.0, 0.0, 15.0, 0.0])
    fixed5 = [0, 2, 4]
    free5 = [1, 3]
    (u5, r5) = fcn(P5, K5, fixed5, free5)
    assert np.allclose(u5[fixed5], 0.0)
    assert np.allclose(r5[free5], 0.0)
    assert np.allclose(P5 + r5, K5 @ u5)
    assert np.allclose(np.sum(P5 + r5), 0.0)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError."""
    K1 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 2.0], [0.0, 2.0, 2.0]])
    P1 = np.array([5.0, 10.0, 10.0])
    fixed1 = [0]
    free1 = [1, 2]
    with pytest.raises(ValueError):
        fcn(P1, K1, fixed1, free1)
    eps = 1e-20
    K2 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 1.0, 0.0], [0.0, 1.0, 1.0 + eps, 0.0], [0.0, 0.0, 0.0, 1.0]])
    P2 = np.array([1.0, 2.0, 3.0, 4.0])
    fixed2 = [0, 3]
    free2 = [1, 2]
    with pytest.raises(ValueError):
        fcn(P2, K2, fixed2, free2)
    K3 = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 0.0]])
    P3 = np.array([1.0, 2.0, 3.0])
    fixed3 = [0]
    free3 = [1, 2]
    with pytest.raises(ValueError):
        fcn(P3, K3, fixed3, free3)
    n = 10
    K4 = np.eye(n)
    K4[5:8, 5:8] = np.ones((3, 3)) * 1e-18
    np.fill_diagonal(K4[5:8, 5:8], 1e-18)
    P4 = np.ones(n)
    fixed4 = list(range(5)) + list(range(8, n))
    free4 = [5, 6, 7]
    with pytest.raises(ValueError):
        fcn(P4, K4, fixed4, free4)
    scale = 10000000000.0
    K5 = np.array([[scale, 0.0, 0.0], [0.0, scale, scale], [0.0, scale, scale]])
    P5 = np.array([scale, 2 * scale, 2 * scale])
    fixed5 = [0]
    free5 = [1, 2]
    with pytest.raises(ValueError):
        fcn(P5, K5, fixed5, free5)