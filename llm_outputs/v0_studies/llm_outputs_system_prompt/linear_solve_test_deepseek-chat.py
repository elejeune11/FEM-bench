def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    K1 = np.array([[2, -1], [-1, 2]], dtype=float)
    P1 = np.array([1, 0], dtype=float)
    fixed1 = [0]
    free1 = [1]
    (u1, r1) = fcn(P1, K1, fixed1, free1)
    assert u1[fixed1] == pytest.approx(0.0)
    expected_u1 = np.array([0, 0.5])
    assert u1 == pytest.approx(expected_u1)
    expected_r1 = np.array([-1.5, 0])
    assert r1 == pytest.approx(expected_r1)
    assert P1 + r1 == pytest.approx(np.zeros_like(P1))
    K2 = np.array([[3, -1, -1], [-1, 2, 0], [-1, 0, 2]], dtype=float)
    P2 = np.array([0, 2, 1], dtype=float)
    fixed2 = [0, 2]
    free2 = [1]
    (u2, r2) = fcn(P2, K2, fixed2, free2)
    assert u2[fixed2] == pytest.approx(0.0)
    expected_u2 = np.array([0, 1.0, 0])
    assert u2 == pytest.approx(expected_u2)
    expected_r2 = np.array([-1, 0, -1])
    assert r2 == pytest.approx(expected_r2)
    assert P2 + r2 == pytest.approx(np.zeros_like(P2))
    K3 = np.array([[4, -1, 0, -1], [-1, 4, -1, 0], [0, -1, 4, -1], [-1, 0, -1, 4]], dtype=float)
    P3 = np.array([1, 0, 2, 0], dtype=float)
    fixed3 = [1, 3]
    free3 = [0, 2]
    (u3, r3) = fcn(P3, K3, fixed3, free3)
    assert u3[fixed3] == pytest.approx(0.0)
    assert P3 + r3 == pytest.approx(np.zeros_like(P3))

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError.
    """
    K_singular = np.array([[1, 1], [1, 1]], dtype=float)
    P = np.array([1, 1], dtype=float)
    fixed = [0]
    free = [1]
    with pytest.raises(ValueError):
        fcn(P, K_singular, fixed, free)
    K_ill_conditioned = np.array([[1e-16, 0], [0, 1]], dtype=float)
    P2 = np.array([1, 1], dtype=float)
    fixed2 = [0]
    free2 = [1]
    with pytest.raises(ValueError):
        fcn(P2, K_ill_conditioned, fixed2, free2)
    K_large_ill = np.array([[1, 1e-16, 0], [1e-16, 1e-16, 0], [0, 0, 1]], dtype=float)
    P3 = np.array([1, 1, 1], dtype=float)
    fixed3 = [0, 2]
    free3 = [1]
    with pytest.raises(ValueError):
        fcn(P3, K_large_ill, fixed3, free3)