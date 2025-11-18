def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    K_global = np.array([[2, -1], [-1, 2]])
    P_global = np.array([1, 0])
    fixed = [0]
    free = [1]
    (u, nodal_reaction_vector) = fcn(P_global, K_global, fixed, free)
    expected_u = np.array([0, 0.5])
    expected_reaction = np.array([-0.5, 0])
    assert np.allclose(u, expected_u)
    assert np.allclose(nodal_reaction_vector, expected_reaction)
    K_global = np.array([[3, -1, -1], [-1, 3, -1], [-1, -1, 3]])
    P_global = np.array([0, 1, 0])
    fixed = [0, 2]
    free = [1]
    (u, nodal_reaction_vector) = fcn(P_global, K_global, fixed, free)
    expected_u = np.array([0, 0.5, 0])
    expected_reaction = np.array([0.5, 0, 0.5])
    assert np.allclose(u, expected_u)
    assert np.allclose(nodal_reaction_vector, expected_reaction)
    assert np.allclose(P_global, K_global @ u)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError.
    """
    K_global = np.zeros((2, 2))
    P_global = np.array([1, 0])
    fixed = [0]
    free = [1]
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)
    K_global = np.array([[1e-16, 0], [0, 1e-16]])
    P_global = np.array([1, 0])
    fixed = [0]
    free = [1]
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)