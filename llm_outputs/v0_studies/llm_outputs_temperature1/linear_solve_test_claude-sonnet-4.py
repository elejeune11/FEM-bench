def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations."""
    K_global = np.array([[2.0, -1.0], [-1.0, 2.0]])
    P_global = np.array([0.0, 10.0])
    fixed = [0]
    free = [1]
    (u, reactions) = fcn(P_global, K_global, fixed, free)
    assert u[0] == 0.0
    total_force = K_global @ u
    expected_reactions = total_force - P_global
    assert np.allclose(reactions[fixed], expected_reactions[fixed])
    assert np.allclose(reactions[free], 0.0)
    K_global = np.array([[4.0, -2.0, 0.0], [-2.0, 4.0, -2.0], [0.0, -2.0, 2.0]])
    P_global = np.array([0.0, 5.0, 0.0])
    fixed = [0, 2]
    free = [1]
    (u, reactions) = fcn(P_global, K_global, fixed, free)
    assert u[0] == 0.0
    assert u[2] == 0.0
    total_force = K_global @ u
    expected_reactions = total_force - P_global
    assert np.allclose(reactions[fixed], expected_reactions[fixed])
    assert np.allclose(reactions[free], 0.0)
    K_global = np.array([[3.0, -1.0, 0.0, 0.0], [-1.0, 3.0, -1.0, 0.0], [0.0, -1.0, 3.0, -1.0], [0.0, 0.0, -1.0, 2.0]])
    P_global = np.array([0.0, 8.0, 0.0, 4.0])
    fixed = [0, 2]
    free = [1, 3]
    (u, reactions) = fcn(P_global, K_global, fixed, free)
    assert u[0] == 0.0
    assert u[2] == 0.0
    total_force = K_global @ u
    expected_reactions = total_force - P_global
    assert np.allclose(reactions[fixed], expected_reactions[fixed])
    assert np.allclose(reactions[free], 0.0)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError."""
    K_global = np.array([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0], [1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    P_global = np.array([1.0, 1.0, 1.0, 1.0])
    fixed = [0, 1]
    free = [2, 3]
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)
    eps = 1e-18
    K_global = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, eps]])
    P_global = np.array([0.0, 1.0, 1.0])
    fixed = [0]
    free = [1, 2]
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)