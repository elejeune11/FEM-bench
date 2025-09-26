def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
for small, solvable systems.
Verifies boundary conditions, internal equilibrium, and global force balance across
multiple cases with different free/fixed DOF configurations."""
    K1 = np.array([[100.0, -100.0, 0.0], [-100.0, 300.0, -200.0], [0.0, -200.0, 200.0]])
    P1 = np.array([0.0, 0.0, 10.0])
    fixed1 = [0]
    free1 = [1, 2]
    expected_u1 = np.array([0.0, 0.1, 0.15])
    expected_reaction1 = np.array([-10.0, 0.0, 0.0])
    (u1, r1) = fcn(P1, K1, fixed1, free1)
    assert np.allclose(u1, expected_u1)
    assert np.allclose(r1, expected_reaction1)
    assert np.allclose(K1 @ u1, P1 + r1)
    assert np.allclose(u1[fixed1], 0)
    assert np.allclose(r1[free1], 0)
    P2 = np.array([0.0, 5.0, 0.0])
    fixed2 = [2]
    free2 = [0, 1]
    expected_u2 = np.array([0.025, 0.025, 0.0])
    expected_reaction2 = np.array([0.0, 0.0, -5.0])
    (u2, r2) = fcn(P2, K1, fixed2, free2)
    assert np.allclose(u2, expected_u2)
    assert np.allclose(r2, expected_reaction2)
    assert np.allclose(K1 @ u2, P2 + r2)
    assert np.allclose(u2[fixed2], 0)
    assert np.allclose(r2[free2], 0)
    K3 = np.array([[2.0, -1.0, 0.0, 0.0], [-1.0, 2.0, -1.0, 0.0], [0.0, -1.0, 2.0, -1.0], [0.0, 0.0, -1.0, 1.0]])
    P3 = np.array([0.0, 0.0, 0.0, 10.0])
    fixed3 = [0]
    free3 = [1, 2, 3]
    expected_u3 = np.array([0.0, 10.0, 20.0, 30.0])
    expected_reaction3 = np.array([-10.0, 0.0, 0.0, 0.0])
    (u3, r3) = fcn(P3, K3, fixed3, free3)
    assert np.allclose(u3, expected_u3)
    assert np.allclose(r3, expected_reaction3)
    assert np.allclose(K3 @ u3, P3 + r3)
    assert np.allclose(u3[fixed3], 0)
    assert np.allclose(r3[free3], 0)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
(i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
to a numerically reliable degree.
This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
by using fixed/free DOF partitioning, and checks that the function does not proceed with
solving but instead raises the documented ValueError."""
    K_singular = np.array([[100.0, -100.0], [-100.0, 100.0]])
    P_singular = np.array([10.0, -10.0])
    fixed_singular = []
    free_singular = [0, 1]
    with pytest.raises(ValueError):
        fcn(P_singular, K_singular, fixed_singular, free_singular)
    K_global_singular = np.array([[100, -100, 0], [-100, 300, -200], [0, -200, 200]])
    P_global = np.array([1, 2, 3])
    fixed = []
    free = [0, 1, 2]
    with pytest.raises(ValueError):
        fcn(P_global, K_global_singular, fixed, free)
    K_near_singular = np.array([[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-17, 0.0], [0.0, 0.0, 1.0]])
    P_near_singular = np.ones(3)
    fixed_near_singular = [2]
    free_near_singular = [0, 1]
    with pytest.raises(ValueError):
        fcn(P_near_singular, K_near_singular, fixed_near_singular, free_near_singular)