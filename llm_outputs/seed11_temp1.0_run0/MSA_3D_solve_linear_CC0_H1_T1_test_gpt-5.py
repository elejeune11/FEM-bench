def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    n_nodes_A = 2
    ndof_A = 6 * n_nodes_A
    K_ff_A = np.diag([10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    K_ss_A = np.diag([50.0, 60.0, 70.0, 80.0, 90.0, 100.0])
    K_sf_A = np.array([[2.0, 0.0, 1.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 4.0, 0.0, 2.0, 0.0], [1.0, 0.0, 0.0, 5.0, 0.0, 1.0], [0.0, 1.0, 0.0, 0.0, 6.0, 0.0], [0.0, 0.0, 1.0, 0.0, 0.0, 7.0]])
    K_fs_A = K_sf_A.T
    K_global_A = np.zeros((ndof_A, ndof_A), dtype=float)
    K_global_A[np.ix_(np.arange(6), np.arange(6))] = K_ss_A
    K_global_A[np.ix_(np.arange(6), np.arange(6, 12))] = K_sf_A
    K_global_A[np.ix_(np.arange(6, 12), np.arange(6))] = K_fs_A
    K_global_A[np.ix_(np.arange(6, 12), np.arange(6, 12))] = K_ff_A
    P_fixed_A = np.array([-1.0, 0.5, 2.0, 0.0, -3.0, 4.0])
    P_free_A = np.array([1.0, -2.0, 3.0, -4.0, 5.0, -6.0])
    P_global_A = np.zeros(ndof_A, dtype=float)
    P_global_A[0:6] = P_fixed_A
    P_global_A[6:12] = P_free_A
    boundary_conditions_A = {0: np.array([True, True, True, True, True, True], dtype=bool)}
    (u_A, r_A) = fcn(P_global_A, K_global_A, boundary_conditions_A, n_nodes_A)
    u_free_expected_A = np.linalg.solve(K_ff_A, P_free_A)
    u_expected_A = np.zeros(ndof_A, dtype=float)
    u_expected_A[6:12] = u_free_expected_A
    r_expected_A = np.zeros(ndof_A, dtype=float)
    r_expected_A[0:6] = K_sf_A @ u_free_expected_A - P_fixed_A
    assert u_A.shape == (ndof_A,)
    assert r_A.shape == (ndof_A,)
    assert np.allclose(u_A, u_expected_A, rtol=1e-12, atol=1e-12)
    assert np.allclose(r_A, r_expected_A, rtol=1e-12, atol=1e-12)
    assert np.allclose(K_ff_A @ u_A[6:12], P_free_A, rtol=1e-12, atol=1e-12)
    assert np.allclose(K_global_A @ u_A, P_global_A + r_A, rtol=1e-12, atol=1e-12)
    n_nodes_B = 3
    ndof_B = 6 * n_nodes_B
    rng = np.random.default_rng(12345)
    M = rng.uniform(-1.0, 1.0, size=(ndof_B, ndof_B))
    K_global_B = M.T @ M + 10.0 * np.eye(ndof_B)
    P_global_B = rng.normal(size=ndof_B)
    boundary_conditions_B = {0: np.array([True, True, True, True, True, True], dtype=bool), 1: np.array([True, True, True, False, False, False], dtype=bool)}
    fixed_mask_B = np.zeros(ndof_B, dtype=bool)
    for (node, bc) in boundary_conditions_B.items():
        for j in range(6):
            if bc[j]:
                fixed_mask_B[6 * node + j] = True
    free_mask_B = ~fixed_mask_B
    fixed_idx_B = np.where(fixed_mask_B)[0]
    free_idx_B = np.where(free_mask_B)[0]
    K_ff_B = K_global_B[np.ix_(free_idx_B, free_idx_B)]
    K_sf_B = K_global_B[np.ix_(fixed_idx_B, free_idx_B)]
    P_f_B = P_global_B[free_idx_B]
    P_s_B = P_global_B[fixed_idx_B]
    u_f_expected_B = np.linalg.solve(K_ff_B, P_f_B)
    u_expected_B = np.zeros(ndof_B, dtype=float)
    u_expected_B[free_idx_B] = u_f_expected_B
    r_expected_B = np.zeros(ndof_B, dtype=float)
    r_expected_B[fixed_idx_B] = K_sf_B @ u_f_expected_B - P_s_B
    (u_B, r_B) = fcn(P_global_B, K_global_B, boundary_conditions_B, n_nodes_B)
    assert u_B.shape == (ndof_B,)
    assert r_B.shape == (ndof_B,)
    assert np.allclose(u_B, u_expected_B, rtol=1e-10, atol=1e-10)
    assert np.allclose(r_B, r_expected_B, rtol=1e-10, atol=1e-10)
    assert np.allclose(K_ff_B @ u_B[free_idx_B], P_f_B, rtol=1e-10, atol=1e-10)
    assert np.allclose(r_B[free_idx_B], 0.0, rtol=1e-12, atol=1e-12)
    assert np.allclose(K_global_B @ u_B, P_global_B + r_B, rtol=1e-10, atol=1e-10)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 1
    ndof = 6 * n_nodes
    K_global = np.zeros((ndof, ndof), dtype=float)
    P_global = np.zeros(ndof, dtype=float)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)