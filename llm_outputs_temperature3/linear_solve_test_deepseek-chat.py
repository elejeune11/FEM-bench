def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    K_global = np.array([[2, -1], [-1, 2]], dtype=float)
    P_global = np.array([1, 0], dtype=float)
    fixed = [0]
    free = [1]
    (u, nodal_reaction_vector) = fcn(P_global, K_global, fixed, free)
    assert u[0] == 0.0
    expected_u1 = 0.5
    assert np.isclose(u[1], expected_u1)
    expected_reaction = -1.0
    total_force = P_global + nodal_reaction_vector
    assert np.allclose(total_force, np.zeros_like(total_force), atol=1e-10)
    K_global = np.array([[3, -1, -1], [-1, 2, 0], [-1, 0, 2]], dtype=float)
    P_global = np.array([0, 2, 1], dtype=float)
    fixed = [0]
    free = [1, 2]
    (u, nodal_reaction_vector) = fcn(P_global, K_global, fixed, free)
    assert u[0] == 0.0
    total_force = P_global + nodal_reaction_vector
    assert np.allclose(total_force, np.zeros_like(total_force), atol=1e-10)
    assert nodal_reaction_vector[1] == 0.0
    assert nodal_reaction_vector[2] == 0.0

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError.
    """
    K_global = np.array([[1, 1, 0], [1, 1, 0], [0, 0, 1]], dtype=float)
    P_global = np.array([1, 1, 1], dtype=float)
    fixed = [0]
    free = [1, 2]
    K_global = np.array([[1e-16, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=float)
    P_global = np.array([1, 1, 1], dtype=float)
    fixed = [1, 2]
    free = [0]
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)
    K_global = np.array([[1, 1], [1, 1 + 1e-15]], dtype=float)
    P_global = np.array([1, 1], dtype=float)
    fixed = []
    free = [0, 1]
    with pytest.raises(ValueError):
        fcn(P_global, K_global, fixed, free)