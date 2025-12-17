def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes = 2
    dof_per_node = 6
    n_dofs = dof_per_node * n_nodes
    K_ss_A = np.diag([100.0, 120.0, 130.0, 140.0, 150.0, 160.0])
    K_ff_A = np.diag([2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
    K_sf_A = np.zeros((6, 6))
    K_global_A = np.zeros((n_dofs, n_dofs), dtype=float)
    K_global_A[:6, :6] = K_ss_A
    K_global_A[:6, 6:] = K_sf_A
    K_global_A[6:, :6] = K_sf_A.T
    K_global_A[6:, 6:] = K_ff_A
    P_s_A = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    P_f_A = np.array([10.0, 9.0, 8.0, 7.0, 6.0, 5.0])
    P_global_A = np.zeros(n_dofs, dtype=float)
    P_global_A[:6] = P_s_A
    P_global_A[6:] = P_f_A
    boundary_conditions_A = {0: np.array([True, True, True, True, True, True], dtype=bool)}
    u_A, r_A = fcn(P_global_A, K_global_A, boundary_conditions_A, n_nodes)
    fixed_A = []
    for node in range(n_nodes):
        bc_node = boundary_conditions_A.get(node, [False] * dof_per_node)
        for j in range(dof_per_node):
            if bc_node[j]:
                fixed_A.append(node * dof_per_node + j)
    fixed_A = np.array(fixed_A, dtype=int)
    free_A = np.setdiff1d(np.arange(n_dofs, dtype=int), fixed_A)
    assert u_A.shape == (n_dofs,)
    assert r_A.shape == (n_dofs,)
    assert np.allclose(u_A[fixed_A], 0.0)
    assert np.allclose(r_A[free_A], 0.0)
    K_ff_ex_A = K_global_A[np.ix_(free_A, free_A)]
    K_sf_ex_A = K_global_A[np.ix_(fixed_A, free_A)]
    assert np.allclose(K_ff_ex_A @ u_A[free_A], P_global_A[free_A], atol=1e-12, rtol=0)
    expected_r_fixed_A = K_sf_ex_A @ u_A[free_A] - P_global_A[fixed_A]
    assert np.allclose(r_A[fixed_A], expected_r_fixed_A, atol=1e-12, rtol=0)
    assert np.allclose(K_global_A @ u_A - r_A, P_global_A, atol=1e-12, rtol=0)
    rng = np.random.default_rng(12345)
    K_ff_B = rng.normal(size=(6, 6))
    K_ff_B = K_ff_B.T @ K_ff_B + 10.0 * np.eye(6)
    K_ss_B = rng.normal(size=(6, 6))
    K_ss_B = K_ss_B.T @ K_ss_B + 20.0 * np.eye(6)
    K_sf_B = rng.normal(size=(6, 6))
    K_global_B = np.zeros((n_dofs, n_dofs), dtype=float)
    K_global_B[:6, :6] = K_ss_B
    K_global_B[:6, 6:] = K_sf_B
    K_global_B[6:, :6] = K_sf_B.T
    K_global_B[6:, 6:] = K_ff_B
    P_s_B = rng.normal(size=6)
    P_f_B = rng.normal(size=6)
    P_global_B = np.zeros(n_dofs, dtype=float)
    P_global_B[:6] = P_s_B
    P_global_B[6:] = P_f_B
    boundary_conditions_B = {0: np.array([True, True, True, True, True, True], dtype=bool)}
    u_B, r_B = fcn(P_global_B, K_global_B, boundary_conditions_B, n_nodes)
    fixed_B = []
    for node in range(n_nodes):
        bc_node = boundary_conditions_B.get(node, [False] * dof_per_node)
        for j in range(dof_per_node):
            if bc_node[j]:
                fixed_B.append(node * dof_per_node + j)
    fixed_B = np.array(fixed_B, dtype=int)
    free_B = np.setdiff1d(np.arange(n_dofs, dtype=int), fixed_B)
    assert u_B.shape == (n_dofs,)
    assert r_B.shape == (n_dofs,)
    assert np.allclose(u_B[fixed_B], 0.0)
    assert np.allclose(r_B[free_B], 0.0)
    K_ff_ex_B = K_global_B[np.ix_(free_B, free_B)]
    K_sf_ex_B = K_global_B[np.ix_(fixed_B, free_B)]
    assert np.allclose(K_ff_ex_B @ u_B[free_B], P_global_B[free_B], atol=1e-12, rtol=0)
    expected_r_fixed_B = K_sf_ex_B @ u_B[free_B] - P_global_B[fixed_B]
    assert np.allclose(r_B[fixed_B], expected_r_fixed_B, atol=1e-12, rtol=0)
    assert np.allclose(K_global_B @ u_B - r_B, P_global_B, atol=1e-12, rtol=0)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    dof_per_node = 6
    n_dofs = dof_per_node * n_nodes
    K_ss = np.diag([10.0, 11.0, 12.0, 13.0, 14.0, 15.0])
    K_ff = np.zeros((6, 6), dtype=float)
    K_sf = np.zeros((6, 6), dtype=float)
    K_global = np.zeros((n_dofs, n_dofs), dtype=float)
    K_global[:6, :6] = K_ss
    K_global[:6, 6:] = K_sf
    K_global[6:, :6] = K_sf.T
    K_global[6:, 6:] = K_ff
    P_global = np.zeros(n_dofs, dtype=float)
    P_global[:6] = np.array([1.0, 0.0, -1.0, 2.0, -2.0, 3.0])
    P_global[6:] = np.array([0.5, -0.5, 1.5, -1.5, 2.5, -2.5])
    boundary_conditions = {0: np.array([True, True, True, True, True, True], dtype=bool)}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)