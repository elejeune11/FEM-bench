def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
mimic cantilever-style setups. Checks boundary-condition handling,
free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    k_base = 10.0
    K_global = np.eye(n_dof) * k_base
    k_spring = 1000.0
    K_global[0, 0] += k_spring
    K_global[6, 6] += k_spring
    K_global[0, 6] -= k_spring
    K_global[6, 0] -= k_spring
    P_global = np.zeros(n_dof)
    f_applied = 50.0
    P_global[6] = f_applied
    boundary_conditions = {0: [True] * 6}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert np.allclose(u[0:6], 0.0), 'Displacements at fixed DOFs must be zero.'
    expected_u6 = f_applied / (k_base + k_spring)
    assert np.isclose(u[6], expected_u6), 'Displacement at free DOF is incorrect.'
    expected_r0 = -k_spring * expected_u6
    assert np.isclose(r[0], expected_r0), 'Reaction force at fixed support is incorrect.'
    assert np.allclose(r[6:], 0.0), 'Reactions at free DOFs must be zero.'
    forces_internal = K_global @ u
    forces_external = P_global + r
    assert np.allclose(forces_internal, forces_external, atol=1e-10), 'Global equilibrium not satisfied.'

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned."""
    n_nodes = 2
    n_dof = 12
    K_global = np.zeros((n_dof, n_dof))
    P_global = np.zeros(n_dof)
    boundary_conditions = {0: [True] * 6}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)