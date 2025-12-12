def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs))
    k = 1000.0
    for i in range(6):
        K_global[i, i] = 2 * k
        K_global[6 + i, 6 + i] = 2 * k
        K_global[i, 6 + i] = -k
        K_global[6 + i, i] = -k
    P_global = np.zeros(n_dofs)
    P_global[6] = 100.0
    P_global[7] = 50.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    for i in range(6):
        assert np.isclose(u[i], 0.0), f'Fixed DOF {i} should have zero displacement'
    free_dofs = np.arange(6, 12)
    fixed_dofs = np.arange(0, 6)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    u_f = u[free_dofs]
    P_f = P_global[free_dofs]
    equilibrium_residual = K_ff @ u_f - P_f
    assert np.allclose(equilibrium_residual, 0.0, atol=1e-10), 'Free DOF equilibrium not satisfied'
    K_sf = K_global[np.ix_(fixed_dofs, free_dofs)]
    expected_reactions = K_sf @ u_f - P_global[fixed_dofs]
    assert np.allclose(r[fixed_dofs], expected_reactions, atol=1e-10), 'Reactions incorrectly computed'
    assert np.allclose(r[free_dofs], 0.0, atol=1e-10), 'Free DOFs should have zero reactions'
    global_equilibrium = K_global @ u - (P_global + r)
    assert np.allclose(global_equilibrium, 0.0, atol=1e-10), 'Global equilibrium not satisfied'
    boundary_conditions_partial = {0: np.array([True, True, True, False, False, False])}
    (u2, r2) = fcn(P_global, K_global, boundary_conditions_partial, n_nodes)
    for i in range(3):
        assert np.isclose(u2[i], 0.0), f'Fixed DOF {i} should have zero displacement'
    global_equilibrium2 = K_global @ u2 - (P_global + r2)
    assert np.allclose(global_equilibrium2, 0.0, atol=1e-10), 'Global equilibrium not satisfied for partial BCs'

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global = np.zeros((n_dofs, n_dofs))
    for i in range(6):
        K_global[i, i] = 1000.0
    epsilon = 1e-20
    for i in range(6, 12):
        K_global[i, i] = epsilon
        for j in range(6, 12):
            if i != j:
                K_global[i, j] = epsilon * 0.5
    for i in range(6):
        K_global[i, 6 + i] = -100.0
        K_global[6 + i, i] = -100.0
    P_global = np.zeros(n_dofs)
    P_global[6] = 1.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)
    K_global2 = np.eye(n_dofs) * 1e-20
    boundary_conditions_none = {}
    P_global2 = np.zeros(n_dofs)
    P_global2[0] = 1.0
    with pytest.raises(ValueError):
        fcn(P_global2, K_global2, boundary_conditions_none, n_nodes)