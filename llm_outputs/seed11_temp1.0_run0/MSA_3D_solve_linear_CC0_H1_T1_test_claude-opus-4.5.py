def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_base = np.eye(n_dofs) * 1000.0
    for i in range(n_dofs - 1):
        K_base[i, i + 1] = -100.0
        K_base[i + 1, i] = -100.0
    K_base += np.eye(n_dofs) * 500.0
    K_global = K_base
    P_global = np.zeros(n_dofs)
    P_global[6] = 100.0
    P_global[7] = 50.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert u.shape == (n_dofs,), 'Displacement vector has wrong shape'
    assert r.shape == (n_dofs,), 'Reaction vector has wrong shape'
    for i in range(6):
        assert np.isclose(u[i], 0.0), f'Fixed DOF {i} should have zero displacement'
    equilibrium = K_global @ u - P_global - r
    assert np.allclose(equilibrium, 0.0, atol=1e-10), 'Global equilibrium not satisfied'
    for i in range(6, n_dofs):
        assert np.isclose(r[i], 0.0, atol=1e-10), f'Free DOF {i} should have zero reaction'
    free_dofs = list(range(6, n_dofs))
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    u_f = u[free_dofs]
    P_f = P_global[free_dofs]
    K_fs = K_global[np.ix_(free_dofs, list(range(6)))]
    u_s = u[:6]
    rhs = P_f - K_fs @ u_s
    assert np.allclose(K_ff @ u_f, rhs, atol=1e-10), 'Free-DOF equilibrium not satisfied'
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global2 = np.eye(n_dofs) * 2000.0
    for i in range(n_dofs - 1):
        K_global2[i, i + 1] = -50.0
        K_global2[i + 1, i] = -50.0
    P_global2 = np.zeros(n_dofs)
    P_global2[8] = 200.0
    boundary_conditions2 = {0: np.array([True, True, True, False, False, False])}
    (u2, r2) = fcn(P_global2, K_global2, boundary_conditions2, n_nodes)
    for i in range(3):
        assert np.isclose(u2[i], 0.0), f'Fixed DOF {i} should have zero displacement'
    equilibrium2 = K_global2 @ u2 - P_global2 - r2
    assert np.allclose(equilibrium2, 0.0, atol=1e-10), 'Global equilibrium not satisfied for case 2'

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global = np.eye(n_dofs) * 1.0
    K_global[6, 6] = 1.0
    K_global[7, 7] = 1.0
    K_global[6, 7] = 1.0 - 1e-17
    K_global[7, 6] = 1.0 - 1e-17
    K_global[8, 8] = 1e-20
    K_global[9, 9] = 1e+20
    K_global[10, 10] = 1e-20
    K_global[11, 11] = 1e+20
    P_global = np.zeros(n_dofs)
    P_global[6] = 1.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)
    K_global2 = np.eye(n_dofs) * 1000.0
    K_global2[6:12, 6:12] = np.ones((6, 6))
    with pytest.raises(ValueError):
        fcn(P_global, K_global2, boundary_conditions, n_nodes)