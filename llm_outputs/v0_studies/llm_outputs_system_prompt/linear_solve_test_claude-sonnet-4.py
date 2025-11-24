def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations."""
    K_global = np.array([[2.0, -1.0], [-1.0, 2.0]])
    P_global = np.array([0.0, 10.0])
    fixed = np.array([0])
    free = np.array([1])
    (u, reactions) = fcn(P_global, K_global, fixed, free)
    assert u[0] == 0.0
    expected_u_free = 5.0
    assert np.isclose(u[1], expected_u_free)
    expected_reaction = -5.0
    assert np.isclose(reactions[0], expected_reaction)
    assert reactions[1] == 0.0
    K_global = np.array([[4.0, -2.0, 0.0], [-2.0, 4.0, -2.0], [0.0, -2.0, 4.0]])
    P_global = np.array([0.0, 20.0, 0.0])
    fixed = np.array([0, 2])
    free = np.array([1])
    (u, reactions) = fcn(P_global, K_global, fixed, free)
    assert u[0] == 0.0
    assert u[2] == 0.0
    expected_u_free = 5.0
    assert np.isclose(u[1], expected_u_free)
    assert np.isclose(reactions[0], -10.0)
    assert reactions[1] == 0.0
    assert np.isclose(reactions[2], -10.0)
    K_global = np.array([[3.0, -1.0, -1.0], [-1.0, 3.0, -1.0], [-1.0, -1.0, 3.0]])
    P_global = np.array([0.0, 0.0, 15.0])
    fixed = np.array([0, 1])
    free = np.array([2])
    (u, reactions) = fcn(P_global, K_global, fixed, free)
    assert u[0] == 0.0
    assert u[1] == 0.0
    expected_u_free = 5.0
    assert np.isclose(u[2], expected_u_free)
    total_applied = np.sum(P_global)
    total_reactions = np.sum(reactions)
    assert np.isclose(total_applied + total_reactions, 0.0)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError."""
    K_global = np.array([[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    P_global = np.array([10.0, 10.0, 5.0])
    fixed = np.array([2])
    free = np.array([0, 1])
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)
    eps = 1e-18
    K_global = np.array([[1.0, 1.0 - eps, 0.0], [1.0 - eps, 1.0, 0.0], [0.0, 0.0, 1.0]])
    P_global = np.array([10.0, 10.0, 5.0])
    fixed = np.array([2])
    free = np.array([0, 1])
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)