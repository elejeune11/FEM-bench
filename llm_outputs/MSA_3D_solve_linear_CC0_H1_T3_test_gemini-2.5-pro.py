def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes_1 = 2
    dofs_1 = n_nodes_1 * 6
    bc_1 = {0: np.ones(6, dtype=bool)}
    P_1 = np.zeros(dofs_1)
    P_1[7] = -10.0
    K_ff_1 = np.diag([10.0, 20.0, 30.0, 40.0, 50.0, 60.0])
    K_ss_1 = np.diag([2.0, 4.0, 6.0, 8.0, 10.0, 12.0])
    K_sf_1 = -np.eye(6) * 5.0
    K_1 = np.block([[K_ss_1, K_sf_1], [K_sf_1.T, K_ff_1]])
    u_f_1 = np.linalg.solve(K_ff_1, P_1[6:])
    u_1_expected = np.zeros(dofs_1)
    u_1_expected[6:] = u_f_1
    r_s_1 = K_sf_1 @ u_f_1
    r_1_expected = np.zeros(dofs_1)
    r_1_expected[:6] = r_s_1
    (u_actual_1, r_actual_1) = fcn(P_1, K_1, bc_1, n_nodes_1)
    assert np.allclose(u_actual_1, u_1_expected)
    assert np.allclose(r_actual_1, r_1_expected)
    assert np.allclose(K_1 @ u_actual_1, P_1 + r_actual_1)
    n_nodes_2 = 3
    dofs_2 = n_nodes_2 * 6
    bc_2 = {0: np.array([True, True, True, False, False, False]), 2: np.array([False, True, False, False, True, False])}
    np.random.seed(42)
    A_2 = np.random.rand(dofs_2, dofs_2)
    K_2 = A_2.T @ A_2 + np.eye(dofs_2)
    P_2 = np.zeros(dofs_2)
    P_2[7] = 100.0
    P_2[10] = -50.0
    fixed_dofs_2 = []
    free_dofs_2 = []
    for i in range(n_nodes_2):
        bcs_node = bc_2.get(i, np.zeros(6, dtype=bool))
        for j in range(6):
            dof_index = i * 6 + j
            if bcs_node[j]:
                fixed_dofs_2.append(dof_index)
            else:
                free_dofs_2.append(dof_index)
    P_f_2 = P_2[free_dofs_2]
    P_s_2 = P_2[fixed_dofs_2]
    K_ff_2 = K_2[np.ix_(free_dofs_2, free_dofs_2)]
    K_sf_2 = K_2[np.ix_(fixed_dofs_2, free_dofs_2)]
    u_f_2 = np.linalg.solve(K_ff_2, P_f_2)
    r_s_2 = K_sf_2 @ u_f_2 - P_s_2
    u_2_expected = np.zeros(dofs_2)
    u_2_expected[free_dofs_2] = u_f_2
    r_2_expected = np.zeros(dofs_2)
    r_2_expected[fixed_dofs_2] = r_s_2
    (u_actual_2, r_actual_2) = fcn(P_2, K_2, bc_2, n_nodes_2)
    assert np.allclose(u_actual_2, u_2_expected)
    assert np.allclose(r_actual_2, r_2_expected)
    assert np.allclose(K_2 @ u_actual_2, P_2 + r_actual_2)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes_1 = 2
    dofs_1 = n_nodes_1 * 6
    K_1 = np.zeros((dofs_1, dofs_1))
    P_1 = np.zeros(dofs_1)
    bc_1 = {}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(P_1, K_1, bc_1, n_nodes_1)
    n_nodes_2 = 2
    dofs_2 = n_nodes_2 * 6
    K_2 = np.zeros((dofs_2, dofs_2))
    P_2 = np.zeros(dofs_2)
    bc_2 = {0: np.ones(6, dtype=bool)}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(P_2, K_2, bc_2, n_nodes_2)
    n_nodes_3 = 1
    dofs_3 = n_nodes_3 * 6
    K_3 = np.diag([10000000000.0, 1.0, 1.0, 1.0, 1.0, 1e-10])
    P_3 = np.ones(dofs_3)
    bc_3 = {}
    with pytest.raises(ValueError, match='ill-conditioned'):
        fcn(P_3, K_3, bc_3, n_nodes_3)