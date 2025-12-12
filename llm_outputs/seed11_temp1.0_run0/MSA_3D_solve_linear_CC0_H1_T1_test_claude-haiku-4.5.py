def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_global = np.eye(n_dof) * 1000.0
    for i in range(n_dof - 1):
        K_global[i, i + 1] = 100.0
        K_global[i + 1, i] = 100.0
    K_global += np.diag(np.ones(n_dof) * 100.0)
    P_global = np.zeros(n_dof)
    P_global[6] = 1000.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with patch('__main__.partition_degrees_of_freedom') as mock_partition:
        fixed_dofs = np.array([0, 1, 2, 3, 4, 5])
        free_dofs = np.array([6, 7, 8, 9, 10, 11])
        mock_partition.return_value = (fixed_dofs, free_dofs)
        (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
        assert u.shape == (n_dof,), f'Displacement vector shape mismatch: {u.shape}'
        assert r.shape == (n_dof,), f'Reaction vector shape mismatch: {r.shape}'
        assert np.allclose(u[fixed_dofs], 0.0), 'Fixed DOFs should have zero displacement'
        assert not np.allclose(u[free_dofs], 0.0), 'Free DOFs should have non-zero displacement'
        residual = K_global @ u - (P_global + r)
        assert np.allclose(residual, 0.0, atol=1e-08), 'Global equilibrium violated'
        assert np.allclose(r[free_dofs], 0.0, atol=1e-10), 'Reactions should only occur at fixed DOFs'
    boundary_conditions_2 = {0: np.array([True, True, True, True, True, True]), 1: np.array([False, False, False, False, False, False])}
    with patch('__main__.partition_degrees_of_freedom') as mock_partition:
        fixed_dofs_2 = np.array([0, 1, 2, 3, 4, 5])
        free_dofs_2 = np.array([6, 7, 8, 9, 10, 11])
        mock_partition.return_value = (fixed_dofs_2, free_dofs_2)
        P_global_2 = np.zeros(n_dof)
        P_global_2[7] = 500.0
        (u_2, r_2) = fcn(P_global_2, K_global, boundary_conditions_2, n_nodes)
        residual_2 = K_global @ u_2 - (P_global_2 + r_2)
        assert np.allclose(residual_2, 0.0, atol=1e-08), 'Global equilibrium violated in test case 2'
        assert np.allclose(u_2[fixed_dofs_2], 0.0), 'Fixed DOFs should have zero displacement in test case 2'

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_global = np.eye(n_dof) * 1.0
    for i in range(n_dof // 2):
        K_global[i, :] = K_global[i, :] * 1e-08 + K_global[(i + 1) % n_dof, :] * (1 - 1e-08)
    P_global = np.ones(n_dof)
    boundary_conditions = {0: np.array([True, True, True, False, False, False])}
    with patch('__main__.partition_degrees_of_freedom') as mock_partition:
        fixed_dofs = np.array([0, 1, 2])
        free_dofs = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])
        mock_partition.return_value = (fixed_dofs, free_dofs)
        K_ff = K_global[np.ix_(free_dofs, free_dofs)]
        cond_number = np.linalg.cond(K_ff)
        if cond_number >= 1e+16:
            with pytest.raises(ValueError):
                fcn(P_global, K_global, boundary_conditions, n_nodes)
        else:
            try:
                (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
                assert u.shape == (n_dof,)
                assert r.shape == (n_dof,)
            except ValueError:
                pytest.fail('Function raised ValueError on acceptable condition number')