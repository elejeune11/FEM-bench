def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_global = np.eye(n_dof) * 1000.0
    K_global[0:3, 0:3] = np.array([[2000, 100, 50], [100, 2000, 75], [50, 75, 2000]])
    K_global[6:9, 6:9] = np.array([[2000, 100, 50], [100, 2000, 75], [50, 75, 2000]])
    K_global[0:3, 6:9] = np.array([[-500, -50, -25], [-50, -500, -40], [-25, -40, -500]])
    K_global[6:9, 0:3] = K_global[0:3, 6:9].T
    K_global = (K_global + K_global.T) / 2
    P_global = np.zeros(n_dof)
    P_global[6] = 100.0
    P_global[7] = 50.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert u.shape == (n_dof,), f'Displacement vector shape mismatch: {u.shape}'
    assert r.shape == (n_dof,), f'Reaction vector shape mismatch: {r.shape}'
    assert np.allclose(u[0:6], 0.0, atol=1e-10), 'Fixed node displacements should be zero'
    equilibrium_check = K_global @ u - P_global - r
    assert np.allclose(equilibrium_check, 0.0, atol=1e-08), 'Global equilibrium not satisfied'
    assert np.allclose(r[6:12], 0.0, atol=1e-10), 'Reactions should be zero at free DOFs'
    n_nodes = 1
    n_dof = 6
    K_global = np.eye(n_dof) * 1000.0
    P_global = np.array([100.0, 50.0, 25.0, 0.0, 0.0, 0.0])
    boundary_conditions = {0: np.array([False, True, False, True, True, True])}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert np.allclose(u[1], 0.0, atol=1e-10), 'Constrained DOF should be zero'
    assert np.allclose(u[3:6], 0.0, atol=1e-10), 'Constrained rotations should be zero'
    equilibrium_check = K_global @ u - P_global - r
    assert np.allclose(equilibrium_check, 0.0, atol=1e-08), 'Global equilibrium not satisfied'
    free_dofs = np.array([0, 2])
    fixed_dofs = np.array([1, 3, 4, 5])
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    P_f = P_global[free_dofs]
    u_f = u[free_dofs]
    free_equilibrium = K_ff @ u_f - P_f
    assert np.allclose(free_equilibrium, 0.0, atol=1e-08), 'Free DOF equilibrium not satisfied'

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_global = np.eye(n_dof) * 1.0
    K_global[6:9, 6:9] = np.array([[1e-16, 1e-16, 1e-16], [1e-16, 1e-16, 1e-16], [1e-16, 1e-16, 1e-16]])
    K_global[6:9, 6:9] += np.eye(3) * 1e-17
    P_global = np.zeros(n_dof)
    P_global[6] = 1.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)