def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    np.random.seed(42)
    A = np.random.rand(n_dofs, n_dofs)
    K_global = A @ A.T + 10 * np.eye(n_dofs)
    P_global = np.zeros(n_dofs)
    P_global[6] = 100.0
    P_global[7] = 50.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    fixed_dofs = list(range(6))
    for dof in fixed_dofs:
        assert np.isclose(u[dof], 0.0), f'Fixed DOF {dof} should have zero displacement'
    residual = K_global @ u - P_global - r
    assert np.allclose(residual, 0.0, atol=1e-10), 'Global equilibrium not satisfied'
    free_dofs = list(range(6, 12))
    for dof in free_dofs:
        assert np.isclose(r[dof], 0.0), f'Free DOF {dof} should have zero reaction'
    n_nodes = 2
    n_dofs = 6 * n_nodes
    A = np.random.rand(n_dofs, n_dofs)
    K_global = A @ A.T + 20 * np.eye(n_dofs)
    P_global = np.zeros(n_dofs)
    P_global[8] = 75.0
    boundary_conditions = {0: np.array([True, True, True, False, False, False])}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    for dof in [0, 1, 2]:
        assert np.isclose(u[dof], 0.0), f'Fixed DOF {dof} should have zero displacement'
    residual = K_global @ u - P_global - r
    assert np.allclose(residual, 0.0, atol=1e-10), 'Global equilibrium not satisfied'
    n_nodes = 3
    n_dofs = 6 * n_nodes
    A = np.random.rand(n_dofs, n_dofs)
    K_global = A @ A.T + 30 * np.eye(n_dofs)
    P_global = np.zeros(n_dofs)
    P_global[12] = 200.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True]), 1: np.array([False, True, False, False, False, False])}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert np.isclose(u[7], 0.0), 'Fixed DOF at node 1 should be zero'
    for dof in range(6):
        assert np.isclose(u[dof], 0.0), f'Fixed DOF {dof} at node 0 should be zero'
    residual = K_global @ u - P_global - r
    assert np.allclose(residual, 0.0, atol=1e-10), 'Global equilibrium not satisfied'

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global = np.eye(n_dofs)
    K_ff_block = np.ones((6, 6)) * 1e-20
    K_ff_block[0, 0] = 1.0
    K_global[6:12, 6:12] = K_ff_block
    P_global = np.zeros(n_dofs)
    P_global[6] = 1.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)
    K_global2 = np.eye(n_dofs) * 1e-20
    K_global2[0:6, 0:6] = np.eye(6)
    with pytest.raises(ValueError):
        fcn(P_global, K_global2, boundary_conditions, n_nodes)
    K_global3 = np.eye(n_dofs)
    K_ff_singular = np.array([[1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1], [1, 1, 1, 1, 1, 1]], dtype=float)
    K_global3[6:12, 6:12] = K_ff_singular
    with pytest.raises(ValueError):
        fcn(P_global, K_global3, boundary_conditions, n_nodes)