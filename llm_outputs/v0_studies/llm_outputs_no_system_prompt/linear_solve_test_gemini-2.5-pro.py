def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Tests that the linear solver produces correct displacements and reaction forces
for small, solvable systems.
Verifies boundary conditions, internal equilibrium, and global force balance across
multiple cases with different free/fixed DOF configurations."""
    P_global_1 = np.array([0.0, 10.0])
    K_global_1 = np.array([[2.0, -2.0], [-2.0, 2.0]])
    fixed_1 = [0]
    free_1 = [1]
    u_expected_1 = np.array([0.0, 5.0])
    reaction_expected_1 = np.array([-10.0, 0.0])
    (u_1, nodal_reaction_vector_1) = fcn(P_global_1, K_global_1, fixed_1, free_1)
    assert np.allclose(u_1, u_expected_1)
    assert np.allclose(nodal_reaction_vector_1, reaction_expected_1)
    assert np.allclose(K_global_1 @ u_1, P_global_1 + nodal_reaction_vector_1)
    P_global_2 = np.array([0.0, 10.0, 0.0])
    K_global_2 = np.array([[2.0, -1.0, 0.0], [-1.0, 2.0, -1.0], [0.0, -1.0, 1.0]])
    fixed_2 = [0, 2]
    free_2 = [1]
    u_expected_2 = np.array([0.0, 5.0, 0.0])
    reaction_expected_2 = np.array([-5.0, 0.0, -5.0])
    (u_2, nodal_reaction_vector_2) = fcn(P_global_2, K_global_2, fixed_2, free_2)
    assert np.allclose(u_2, u_expected_2)
    assert np.allclose(nodal_reaction_vector_2, reaction_expected_2)
    assert np.allclose(u_2[fixed_2], 0.0)
    assert np.allclose(nodal_reaction_vector_2[free_2], 0.0)
    assert np.allclose(K_global_2 @ u_2, P_global_2 + nodal_reaction_vector_2)
    P_global_3 = np.array([0.0, 10.0, 0.0])
    K_global_3 = K_global_2
    fixed_3 = [0]
    free_3 = [1, 2]
    u_expected_3 = np.array([0.0, 10.0, 10.0])
    reaction_expected_3 = np.array([-10.0, 0.0, 0.0])
    (u_3, nodal_reaction_vector_3) = fcn(P_global_3, K_global_3, fixed_3, free_3)
    assert np.allclose(u_3, u_expected_3)
    assert np.allclose(nodal_reaction_vector_3, reaction_expected_3)
    assert np.allclose(K_global_3 @ u_3, P_global_3 + nodal_reaction_vector_3)
    P_global_4 = np.array([10.0, 20.0])
    K_global_4 = np.array([[5.0, 2.0], [2.0, 4.0]])
    fixed_4 = [0, 1]
    free_4 = []
    u_expected_4 = np.array([0.0, 0.0])
    reaction_expected_4 = np.array([-10.0, -20.0])
    (u_4, nodal_reaction_vector_4) = fcn(P_global_4, K_global_4, fixed_4, free_4)
    assert np.allclose(u_4, u_expected_4)
    assert np.allclose(nodal_reaction_vector_4, reaction_expected_4)
    assert np.allclose(K_global_4 @ u_4, P_global_4 + nodal_reaction_vector_4)

def test_linear_solve_raises_on_ill_conditioned_matrix(fcn):
    """Verifies that `linear_solve` raises a ValueError when the submatrix `K_ff` is ill-conditioned
(i.e., its condition number exceeds 1e16), indicating that the linear system is not solvable
to a numerically reliable degree.
This test passes a deliberately singular (non-invertible) or nearly singular `K_ff` matrix
by using fixed/free DOF partitioning, and checks that the function does not proceed with
solving but instead raises the documented ValueError."""
    P_global_1 = np.array([10.0, -10.0])
    K_global_1 = np.array([[1.0, -1.0], [-1.0, 1.0]])
    fixed_1 = []
    free_1 = [0, 1]
    with pytest.raises(ValueError):
        fcn(P_global_1, K_global_1, fixed_1, free_1)
    P_global_2 = np.array([1.0, 1.0, 1.0])
    K_global_2 = np.array([[1.0, 1.0, 0.0], [1.0, 1.0 + 1e-17, 0.0], [0.0, 0.0, 1.0]])
    fixed_2 = [2]
    free_2 = [0, 1]
    with pytest.raises(ValueError):
        fcn(P_global_2, K_global_2, fixed_2, free_2)