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
    K_global = K_base + np.eye(n_dofs) * 500.0
    P_global = np.zeros(n_dofs)
    P_global[6] = 100.0
    P_global[7] = 50.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    for (node_idx, bc) in boundary_conditions.items():
        for (local_dof, is_fixed) in enumerate(bc):
            if is_fixed:
                global_dof = node_idx * 6 + local_dof
                assert np.isclose(u[global_dof], 0.0, atol=1e-10), f'Fixed DOF {global_dof} should have zero displacement'
    fixed_dofs = []
    free_dofs = []
    for node_idx in range(n_nodes):
        bc = boundary_conditions.get(node_idx, np.array([False] * 6))
        for local_dof in range(6):
            global_dof = node_idx * 6 + local_dof
            if bc[local_dof]:
                fixed_dofs.append(global_dof)
            else:
                free_dofs.append(global_dof)
    fixed_dofs = np.array(fixed_dofs)
    free_dofs = np.array(free_dofs)
    K_ff = K_global[np.ix_(free_dofs, free_dofs)]
    u_f = u[free_dofs]
    P_f = P_global[free_dofs]
    residual = K_ff @ u_f - P_f
    assert np.allclose(residual, 0.0, atol=1e-08), 'Free-DOF equilibrium K_ff @ u_f = P_f not satisfied'
    global_residual = K_global @ u - (P_global + r)
    assert np.allclose(global_residual, 0.0, atol=1e-08), 'Global equilibrium K @ u = P + r not satisfied'
    for dof in free_dofs:
        assert np.isclose(r[dof], 0.0, atol=1e-10), f'Free DOF {dof} should have zero reaction'
    n_nodes_2 = 3
    n_dofs_2 = 6 * n_nodes_2
    K_base_2 = np.eye(n_dofs_2) * 2000.0
    for i in range(n_dofs_2 - 1):
        K_base_2[i, i + 1] = -150.0
        K_base_2[i + 1, i] = -150.0
    K_global_2 = K_base_2 + np.eye(n_dofs_2) * 800.0
    P_global_2 = np.zeros(n_dofs_2)
    P_global_2[12] = 200.0
    boundary_conditions_2 = {0: np.array([True, True, True, True, True, True]), 1: np.array([True, True, True, False, False, False])}
    (u_2, r_2) = fcn(P_global_2, K_global_2, boundary_conditions_2, n_nodes_2)
    for (node_idx, bc) in boundary_conditions_2.items():
        for (local_dof, is_fixed) in enumerate(bc):
            if is_fixed:
                global_dof = node_idx * 6 + local_dof
                assert np.isclose(u_2[global_dof], 0.0, atol=1e-10)
    global_residual_2 = K_global_2 @ u_2 - (P_global_2 + r_2)
    assert np.allclose(global_residual_2, 0.0, atol=1e-08)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_global = np.eye(n_dofs) * 1.0
    K_global[6:12, 6:12] = np.array([[1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 0.0, 1.0, 1e-20]])
    P_global = np.zeros(n_dofs)
    P_global[6] = 100.0
    boundary_conditions = {0: np.array([True, True, True, True, True, True])}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)
    K_global_2 = np.zeros((n_dofs, n_dofs))
    K_global_2[0:6, 0:6] = np.eye(6) * 1000.0
    P_global_2 = np.zeros(n_dofs)
    P_global_2[6] = 100.0
    boundary_conditions_2 = {0: np.array([True, True, True, True, True, True])}
    with pytest.raises(ValueError):
        fcn(P_global_2, K_global_2, boundary_conditions_2, n_nodes)