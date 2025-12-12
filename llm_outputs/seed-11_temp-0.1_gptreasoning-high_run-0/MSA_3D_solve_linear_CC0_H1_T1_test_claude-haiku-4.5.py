def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global = np.eye(n_dofs) * 1000.0
    K_global[0:6, 0:6] = np.eye(6) * 2000.0
    K_global[6:12, 6:12] = np.eye(6) * 1000.0
    K_global[0:6, 6:12] = -np.eye(6) * 500.0
    K_global[6:12, 0:6] = -np.eye(6) * 500.0
    K_global = (K_global + K_global.T) / 2.0
    P_global = np.zeros(n_dofs)
    P_global[6] = 100.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with patch('__main__.partition_degrees_of_freedom') as mock_partition:
        fixed_dofs = np.array([0, 1, 2, 3, 4, 5])
        free_dofs = np.array([6, 7, 8, 9, 10, 11])
        mock_partition.return_value = (fixed_dofs, free_dofs)
        (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert u.shape == (n_dofs,), 'Displacement vector has incorrect shape'
    assert r.shape == (n_dofs,), 'Reaction vector has incorrect shape'
    assert np.allclose(u[0:6], 0.0), 'Fixed DOFs should have zero displacement'
    assert not np.allclose(u[6:12], 0.0), 'Free DOFs under load should be non-zero'
    assert not np.allclose(r[0:6], 0.0), 'Reactions should exist at fixed DOFs'
    assert np.allclose(r[6:12], 0.0), 'No reactions should exist at free DOFs'
    equilibrium_check = K_global @ u - P_global - r
    assert np.allclose(equilibrium_check, 0.0, atol=1e-10), 'Global equilibrium not satisfied'

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global = np.eye(n_dofs) * 1e-15
    K_global[6:12, 6:12] = np.eye(6) * 1e-15
    P_global = np.zeros(n_dofs)
    P_global[6] = 1.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with patch('__main__.partition_degrees_of_freedom') as mock_partition:
        fixed_dofs = np.array([0, 1, 2, 3, 4, 5])
        free_dofs = np.array([6, 7, 8, 9, 10, 11])
        mock_partition.return_value = (fixed_dofs, free_dofs)
        with pytest.raises(ValueError, match='ill-conditioned|singular'):
            fcn(P_global, K_global, boundary_conditions, n_nodes)

def test_linear_solve_partial_constraints(fcn):
    """
    Verifies correct handling of partially constrained nodes where only some DOFs are fixed.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global = np.eye(n_dofs) * 1000.0
    K_global[0:6, 0:6] = np.eye(6) * 2000.0
    K_global[6:12, 6:12] = np.eye(6) * 1000.0
    K_global[0:6, 6:12] = -np.eye(6) * 500.0
    K_global[6:12, 0:6] = -np.eye(6) * 500.0
    K_global = (K_global + K_global.T) / 2.0
    P_global = np.zeros(n_dofs)
    P_global[6] = 100.0
    boundary_conditions = {0: np.array([True, True, True, False, False, False])}
    with patch('__main__.partition_degrees_of_freedom') as mock_partition:
        fixed_dofs = np.array([0, 1, 2])
        free_dofs = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])
        mock_partition.return_value = (fixed_dofs, free_dofs)
        (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert np.allclose(u[fixed_dofs], 0.0), 'Fixed DOFs should be zero'
    assert u.shape == (n_dofs,), 'Displacement vector shape incorrect'
    assert r.shape == (n_dofs,), 'Reaction vector shape incorrect'

def test_linear_solve_zero_load(fcn):
    """
    Verifies that zero external load results in zero displacements and reactions.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global = np.eye(n_dofs) * 1000.0
    K_global[0:6, 0:6] = np.eye(6) * 2000.0
    K_global[6:12, 6:12] = np.eye(6) * 1000.0
    K_global[0:6, 6:12] = -np.eye(6) * 500.0
    K_global[6:12, 0:6] = -np.eye(6) * 500.0
    K_global = (K_global + K_global.T) / 2.0
    P_global = np.zeros(n_dofs)
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with patch('__main__.partition_degrees_of_freedom') as mock_partition:
        fixed_dofs = np.array([0, 1, 2, 3, 4, 5])
        free_dofs = np.array([6, 7, 8, 9, 10, 11])
        mock_partition.return_value = (fixed_dofs, free_dofs)
        (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert np.allclose(u, 0.0), 'Zero load should result in zero displacements'
    assert np.allclose(r, 0.0), 'Zero load should result in zero reactions'

def test_linear_solve_reaction_computation(fcn):
    """
    Verifies that reactions are correctly computed as r_fixed = K_sf @ u_free - P_fixed.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global = np.eye(n_dofs) * 1000.0
    K_global[0:6, 0:6] = np.eye(6) * 2000.0
    K_global[6:12, 6:12] = np.eye(6) * 1000.0
    K_global[0:6, 6:12] = -np.eye(6) * 500.0
    K_global[6:12, 0:6] = -np.eye(6) * 500.0
    K_global = (K_global + K_global.T) / 2.0
    P_global = np.zeros(n_dofs)
    P_global[6] = 100.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with patch('__main__.partition_degrees_of_freedom') as mock_partition:
        fixed_dofs = np.array([0, 1, 2, 3, 4, 5])
        free_dofs = np.array([6, 7, 8, 9, 10, 11])
        mock_partition.return_value = (fixed_dofs, free_dofs)
        (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    K_sf = K_global[np.ix_(fixed_dofs, free_dofs)]
    u_free = u[free_dofs]
    P_fixed = P_global[fixed_dofs]
    expected_r_fixed = K_sf @ u_free - P_fixed
    assert np.allclose(r[fixed_dofs], expected_r_fixed, atol=1e-10), 'Reactions not computed correctly'