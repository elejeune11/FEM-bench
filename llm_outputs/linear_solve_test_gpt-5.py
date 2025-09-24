def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium (K @ u - P equals reaction),
    and global force balance across multiple cases with different free/fixed DOF configurations.
    """
    K1 = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=float)
    P1 = np.array([0.0, 10.0], dtype=float)
    fixed1 = [0]
    free1 = [1]
    (u1, r1) = fcn(P1, K1, fixed1, free1)
    expected_u1 = np.array([0.0, 5.0])
    expected_r1 = np.array([-5.0, 0.0])
    assert u1.shape == (2,)
    assert r1.shape == (2,)
    assert np.allclose(u1, expected_u1)
    assert np.allclose(r1, expected_r1)
    assert np.allclose(K1 @ u1 - P1, r1)
    assert np.allclose(u1[fixed1], 0.0)
    assert np.allclose(r1[free1], 0.0)
    K2 = np.array([[4.0, -1.0, 0.0], [-1.0, 4.0, -1.0], [0.0, -1.0, 3.0]], dtype=float)
    P2 = np.array([0.0, 2.0, 1.0], dtype=float)
    fixed2 = [0]
    free2 = [1, 2]
    (u2, r2) = fcn(P2, K2, fixed2, free2)
    expected_u2 = np.array([0.0, 7.0 / 11.0, 6.0 / 11.0])
    expected_r2 = np.array([-7.0 / 11.0, 0.0, 0.0])
    assert u2.shape == (3,)
    assert r2.shape == (3,)
    assert np.allclose(u2, expected_u2)
    assert np.allclose(r2, expected_r2)
    assert np.allclose(K2 @ u2 - P2, r2)
    assert np.allclose(u2[fixed2], 0.0)
    assert np.allclose(r2[free2], 0.0)
    K3 = np.array([[2.0, 0.0], [0.0, 3.0]], dtype=float)
    P3 = np.array([4.0, 9.0], dtype=float)
    fixed3 = []
    free3 = [0, 1]
    (u3, r3) = fcn(P3, K3, fixed3, free3)
    expected_u3 = np.array([2.0, 3.0])
    expected_r3 = np.zeros(2)
    assert u3.shape == (2,)
    assert r3.shape == (2,)
    assert np.allclose(u3, expected_u3)
    assert np.allclose(r3, expected_r3)
    assert np.allclose(K3 @ u3 - P3, r3)
    if fixed3:
        assert np.allclose(u3[fixed3], 0.0)
    assert np.allclose(r3[free3], 0.0)
    K4 = np.array([[10.0, 2.0, 0.0], [2.0, 5.0, 1.0], [0.0, 1.0, 3.0]], dtype=float)
    P4 = np.array([1.0, 2.0, 3.0], dtype=float)
    fixed4 = [0, 1]
    free4 = [2]
    (u4, r4) = fcn(P4, K4, fixed4, free4)
    expected_u4 = np.array([0.0, 0.0, 1.0])
    expected_r4 = np.array([-1.0, -1.0, 0.0])
    assert u4.shape == (3,)
    assert r4.shape == (3,)
    assert np.allclose(u4, expected_u4)
    assert np.allclose(r4, expected_r4)
    assert np.allclose(K4 @ u4 - P4, r4)
    assert np.allclose(u4[fixed4], 0.0)
    assert np.allclose(r4[free4], 0.0)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """
    Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (condition number exceeds 1e16). Uses a singular K_ff by choosing all DOFs as free, resulting
    in a rank-deficient system that is not solvable to a numerically reliable degree.
    """
    K = np.array([[1.0, 1.0], [1.0, 1.0]], dtype=float)
    P = np.array([1.0, 1.0], dtype=float)
    fixed = []
    free = [0, 1]
    with pytest.raises(ValueError):
        fcn(P, K, fixed, free)