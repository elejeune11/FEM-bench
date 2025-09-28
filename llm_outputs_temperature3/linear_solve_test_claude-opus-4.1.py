def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations."""
    K1 = np.array([[2.0, -1.0], [-1.0, 2.0]])
    P1 = np.array([0.0, 10.0])
    fixed1 = [0]
    free1 = [1]
    (u1, R1) = fcn(P1, K1, fixed1, free1)
    assert u1[0] == 0.0
    assert np.allclose(u1[1], 5.0)
    assert R1[1] == 0.0
    assert np.allclose(R1[0], -5.0)
    assert np.allclose(K1 @ u1 + R1, P1)
    K2 = np.array([[4.0, -2.0, -1.0], [-2.0, 5.0, -2.0], [-1.0, -2.0, 3.0]])
    P2 = np.array([0.0, 15.0, 0.0])
    fixed2 = [0, 2]
    free2 = [1]
    (u2, R2) = fcn(P2, K2, fixed2, free2)
    assert u2[0] == 0.0
    assert u2[2] == 0.0
    assert np.allclose(u2[1], 3.0)
    assert R2[1] == 0.0
    assert np.allclose(R2[0], -6.0)
    assert np.allclose(R2[2], -6.0)
    assert np.allclose(K2 @ u2 + R2, P2)
    K3 = np.array([[10.0, -3.0, -2.0, 0.0], [-3.0, 8.0, -2.0, -1.0], [-2.0, -2.0, 9.0, -3.0], [0.0, -1.0, -3.0, 7.0]])
    P3 = np.array([0.0, 20.0, 10.0, 0.0])
    fixed3 = [0, 3]
    free3 = [1, 2]
    (u3, R3) = fcn(P3, K3, fixed3, free3)
    assert u3[0] == 0.0
    assert u3[3] == 0.0
    assert R3[1] == 0.0
    assert R3[2] == 0.0
    K_ff = K3[np.ix_(free3, free3)]
    P_f = P3[free3]
    u_f_expected = np.linalg.solve(K_ff, P_f)
    assert np.allclose(u3[free3], u_f_expected)
    K_xf = K3[np.ix_(fixed3, free3)]
    R_x_expected = -K_xf @ u3[free3]
    assert np.allclose(R3[fixed3], R_x_expected)
    assert np.allclose(K3 @ u3 + R3, P3)
    K4 = np.array([[2.0, -1.0], [-1.0, 2.0]])
    P4 = np.array([5.0, 10.0])
    fixed4 = []
    free4 = [0, 1]
    (u4, R4) = fcn(P4, K4, fixed4, free4)
    u_expected = np.linalg.solve(K4, P4)
    assert np.allclose(u4, u_expected)
    assert np.allclose(R4, np.zeros(2))
    assert np.allclose(K4 @ u4, P4)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError."""
    K1 = np.array([[1.0, 0.0, 0.0], [0.0, 2.0, 2.0], [0.0, 2.0, 2.0]])
    P1 = np.array([0.0, 5.0, 5.0])
    fixed1 = [0]
    free1 = [1, 2]
    with pytest.raises(ValueError):
        fcn(P1, K1, fixed1, free1)
    epsilon = 1e-20
    K2 = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0 + epsilon]])
    P2 = np.array([1.0, 2.0, 3.0])
    fixed2 = [0]
    free2 = [1, 2]
    with pytest.raises(ValueError):
        fcn(P2, K2, fixed2, free2)
    K3 = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1e+16, 1e+16, 0.0], [0.0, 1e+16, 1e+16, 0.0], [0.0, 0.0, 0.0, 1.0]])
    P3 = np.array([1.0, 1.0, 1.0, 1.0])
    fixed3 = [0, 3]
    free3 = [1, 2]
    with pytest.raises(ValueError):
        fcn(P3, K3, fixed3, free3)