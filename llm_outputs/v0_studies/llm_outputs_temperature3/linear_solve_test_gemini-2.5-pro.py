@pytest.mark.parametrize('P_global, K_global, fixed, free, u_expected, reaction_expected', [case1, case2, case3, case4])
def test_linear_solve_arbitrary_solvable_cases(fcn, P_global, K_global, fixed, free, u_expected, reaction_expected):
    """
    Tests that the linear solver produces correct displacements and reaction forces
    for small, solvable systems.
    Verifies boundary conditions, internal equilibrium, and global force balance across
    multiple cases with different free/fixed DOF configurations.
    """
    (u_actual, reaction_actual) = fcn(P_global, K_global, fixed, free)
    assert np.allclose(u_actual, u_expected, atol=1e-09)
    assert np.allclose(reaction_actual, reaction_expected, atol=1e-09)
    if fixed:
        assert np.all(u_actual[fixed] == 0.0)
    if free:
        assert np.all(reaction_actual[free] == 0.0)
    internal_forces = K_global @ u_actual
    external_forces = P_global + reaction_actual
    assert np.allclose(internal_forces, external_forces, atol=1e-09)

@pytest.mark.parametrize('P_global, K_global, fixed, free', [ill_case1, ill_case2])
def test_linear_solve_raises_on_ill_conditioned_matrix(fcn, P_global, K_global, fixed, free):
    """
    Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
    (i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
    to a numerically reliable degree.
    This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
    by using fixed/free DOF partitioning, and checks that the function does not proceed with
    solving but instead raises the documented ValueError.
    """
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(P_global, K_global, fixed, free)