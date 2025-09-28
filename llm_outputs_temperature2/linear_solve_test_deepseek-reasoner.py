def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    import numpy as np
    K1 = np.array([[2.0]])
    P1 = np.array([4.0])
    fixed1 = []
    free1 = [0]
    (u1, r1) = fcn(P1, K1, fixed1, free1)
    assert np.allclose(u1, [2.0])
    assert np.allclose(r1, [0.0])
    assert np.allclose(K1 @ u1, P1)
    K2 = np.array([[3.0, -1.0], [-1.0, 2.0]])
    P2 = np.array([5.0, 0.0])
    fixed2 = [1]
    free2 = [0]
    (u2, r2) = fcn(P2, K2, fixed2, free2)
    expected_u2 = np.array([5.0 / 3.0, 0.0])
    expected_r2 = np.array([0.0, -5.0 / 3.0])
    assert np.allclose(u2, expected_u2)
    assert np.allclose(r2, expected_r2)
    assert np.allclose(K2 @ u2 - P2, -r2)
    K3 = np.array([[4.0, -1.0, 0.0], [-1.0, 5.0, -2.0], [0.0, -2.0, 3.0]])
    P3 = np.array([1.0, 2.0, 3.0])
    fixed3 = [0, 2]
    free3 = [1]
    (u3, r3) = fcn(P3, K3, fixed3, free3)
    expected_u3 = np.array([0.0, 0.4, 0.0])
    expected_r3 = np.array([0.4, 0.0, 0.8])
    assert np.allclose(u3, expected_u3)
    assert np.allclose(r3, expected_r3)
    assert np.allclose(K3 @ u3 - P3, -r3)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError.
    """
    import numpy as np
    K_singular = np.array([[1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    P_singular = np.array([1.0, 2.0, 3.0])
    fixed_singular = [0, 2]
    free_singular = [1]
    try:
        fcn(P_singular, K_singular, fixed_singular, free_singular)
        assert False, 'Expected ValueError for singular K_ff matrix'
    except ValueError:
        pass
    K_near_singular = np.array([[1.0, 1.0], [1.0, 1.0 + 1e-17]])
    P_near_singular = np.array([1.0, 1.0])
    fixed_near_singular = []
    free_near_singular = [0, 1]
    try:
        fcn(P_near_singular, K_near_singular, fixed_near_singular, free_near_singular)
        assert False, 'Expected ValueError for ill-conditioned K_ff matrix'
    except ValueError:
        pass
    K_well_conditioned = np.array([[2.0, 0.0], [0.0, 2.0]])
    P_well_conditioned = np.array([1.0, 1.0])
    fixed_well_conditioned = []
    free_well_conditioned = [0, 1]
    (u, r) = fcn(P_well_conditioned, K_well_conditioned, fixed_well_conditioned, free_well_conditioned)
    assert np.allclose(u, [0.5, 0.5])