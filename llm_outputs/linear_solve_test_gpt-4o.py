def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    P_global = np.array([0, 10])
    K_global = np.array([[10, -2], [-2, 4]])
    fixed = [0]
    free = [1]
    u_expected = np.array([0, 2.5])
    reaction_expected = np.array([5, 0])
    (u, reaction) = fcn(P_global, K_global, fixed, free)
    assert np.allclose(u, u_expected), 'Displacement vector is incorrect'
    assert np.allclose(reaction, reaction_expected), 'Reaction vector is incorrect'
    P_global = np.array([0, 0, 15])
    K_global = np.array([[12, -3, 0], [-3, 9, -1], [0, -1, 4]])
    fixed = [0, 1]
    free = [2]
    u_expected = np.array([0, 0, 3.75])
    reaction_expected = np.array([0, 3.75, 0])
    (u, reaction) = fcn(P_global, K_global, fixed, free)
    assert np.allclose(u, u_expected), 'Displacement vector is incorrect'
    assert np.allclose(reaction, reaction_expected), 'Reaction vector is incorrect'

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """
    Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError.
    """
    P_global = np.array([0, 0])
    K_global = np.array([[1, 1], [1, 1]])
    fixed = [0]
    free = [1]
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)
    P_global = np.array([0, 0])
    K_global = np.array([[1, 1], [1, 1 + 1e-17]])
    fixed = [0]
    free = [1]
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)