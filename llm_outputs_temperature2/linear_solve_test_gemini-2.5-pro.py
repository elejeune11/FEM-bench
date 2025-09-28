def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
for small, solvable systems.
Verifies boundary conditions, internal equilibrium, and global force balance across
multiple cases with different free/fixed DOF configurations."""
    K1 = np.array([[3.0, -1.0], [-1.0, 1.0]])
    P1 = np.array([0.0, 10.0])
    fixed1 = [0]
    free1 = [1]
    u_expected1 = np.array([0.0, 10.0])
    reaction_expected1 = np.array([-10.0, 0.0])
    K2 = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    P2 = np.array([0.0, 5.0, 10.0])
    fixed2 = [0]
    free2 = [1, 2]
    u_expected2 = np.array([0.0, 15.0, 25.0])
    reaction_expected2 = np.array([-15.0, 0.0, 0.0])
    K3 = K2
    P3 = P2
    fixed3 = [0, 2]
    free3 = [1]
    u_expected3 = np.array([0.0, 2.5, 0.0])
    reaction_expected3 = np.array([-2.5, 0.0, -12.5])
    cases = [(K1, P1, fixed1, free1, u_expected1, reaction_expected1), (K2, P2, fixed2, free2, u_expected2, reaction_expected2), (K3, P3, fixed3, free3, u_expected3, reaction_expected3)]
    for (K_global, P_global, fixed, free, u_expected, reaction_expected) in cases:
        (u, nodal_reaction_vector) = fcn(P_global, K_global, fixed, free)
        np.testing.assert_allclose(u, u_expected, atol=1e-09)
        np.testing.assert_allclose(nodal_reaction_vector, reaction_expected, atol=1e-09)
        np.testing.assert_allclose(K_global @ u, P_global + nodal_reaction_vector, atol=1e-09)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
(i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
to a numerically reliable degree.
This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
by using fixed/free DOF partitioning, and checks that the function does not proceed with
solving but instead raises the documented ValueError."""
    P_global = np.array([1.0, 2.0, 3.0])
    fixed = [0]
    free = [1, 2]
    ill_conditioned_matrices = [np.array([[10.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0]]), np.array([[10.0, 0.0, 0.0], [0.0, 1.0, 1.0], [0.0, 1.0, 1.0 + 1e-16]])]
    for K_global in ill_conditioned_matrices:
        with pytest.raises(ValueError):
            fcn(P_global, K_global, fixed, free)