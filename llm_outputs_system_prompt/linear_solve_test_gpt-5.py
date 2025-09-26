def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    cases = []
    K1 = np.array([[2.0, -1.0], [-1.0, 2.0]])
    P1 = np.array([1.0, 0.5])
    fixed1 = [1]
    free1 = [0]
    cases.append((K1, P1, fixed1, free1))
    K2 = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 2.0]])
    P2 = np.array([1.0, 0.0, 1.0])
    fixed2 = [1]
    free2 = [0, 2]
    cases.append((K2, P2, fixed2, free2))
    K3 = np.array([[2.0, -1.0, 0.0, 0.0], [-1.0, 2.0, -1.0, 0.0], [0.0, -1.0, 2.0, -1.0], [0.0, 0.0, -1.0, 2.0]])
    P3 = np.array([0.0, 1.0, 1.0, 0.0])
    fixed3 = [0, 3]
    free3 = [1, 2]
    cases.append((K3, P3, fixed3, free3))
    for (K, P, fixed, free) in cases:
        (u, r) = fcn(P, K, fixed, free)
        assert isinstance(u, np.ndarray)
        assert isinstance(r, np.ndarray)
        assert u.shape == P.shape
        assert r.shape == P.shape
        assert np.allclose(u[fixed], 0.0, atol=1e-12, rtol=0.0)
        K_ff = K[np.ix_(free, free)]
        assert np.allclose(K_ff @ u[free], P[free], atol=1e-12, rtol=0.0)
        Ku_minus_P = K @ u - P
        assert np.allclose(r, Ku_minus_P, atol=1e-12, rtol=0.0)
        assert np.allclose(r[free], 0.0, atol=1e-12, rtol=0.0)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """
    Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular `K_ff` matrix by using fixed/free DOF partitioning,
    and checks that the function raises the documented ValueError.
    """
    K = np.array([[1.0, 2.0, 0.0], [2.0, 4.0, 0.0], [0.0, 0.0, 1.0]])
    P = np.array([1.0, 1.0, 0.0])
    fixed = [2]
    free = [0, 1]
    raised = False
    try:
        fcn(P, K, fixed, free)
    except ValueError:
        raised = True
    assert raised