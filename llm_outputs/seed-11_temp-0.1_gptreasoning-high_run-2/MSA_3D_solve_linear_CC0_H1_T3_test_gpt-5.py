def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    rng = np.random.default_rng(42)
    n_nodes = 2
    ndof = 6 * n_nodes
    A = rng.normal(size=(ndof, ndof))
    K_global = A.T @ A + 10.0 * np.eye(ndof)
    P_global = rng.normal(size=ndof)

    def mask_from_bc(boundary_conditions, n_nodes):
        fixed = np.zeros(6 * n_nodes, dtype=bool)
        for node, bc in boundary_conditions.items():
            bc_arr = np.asarray(bc, dtype=bool)
            assert bc_arr.shape == (6,)
            idx0 = 6 * node
            fixed[idx0:idx0 + 6] = bc_arr
        free = ~fixed
        return (fixed, free)
    bc1 = {0: [True, True, True, True, True, True]}
    fixed1, free1 = mask_from_bc(bc1, n_nodes)
    K_ff_1 = K_global[np.ix_(free1, free1)]
    P_f_1 = P_global[free1]
    u_f_expected_1 = np.linalg.solve(K_ff_1, P_f_1)
    u_expected_1 = np.zeros(ndof)
    u_expected_1[free1] = u_f_expected_1
    K_sf_1 = K_global[np.ix_(fixed1, free1)]
    P_s_1 = P_global[fixed1]
    r_expected_1 = np.zeros(ndof)
    r_expected_1[fixed1] = K_sf_1 @ u_f_expected_1 - P_s_1
    u1, r1 = fcn(P_global, K_global, bc1, n_nodes)
    assert u1.shape == (ndof,)
    assert r1.shape == (ndof,)
    assert np.allclose(u1[fixed1], 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(r1[free1], 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(K_ff_1 @ u1[free1], P_f_1, atol=1e-10, rtol=1e-10)
    assert np.allclose(r1[fixed1], r_expected_1[fixed1], atol=1e-10, rtol=1e-10)
    residual1 = K_global @ u1 - P_global - r1
    assert np.allclose(residual1, 0.0, atol=1e-10, rtol=1e-10)
    bc2 = {0: [True, True, True, True, False, False]}
    fixed2, free2 = mask_from_bc(bc2, n_nodes)
    K_ff_2 = K_global[np.ix_(free2, free2)]
    P_f_2 = P_global[free2]
    u_f_expected_2 = np.linalg.solve(K_ff_2, P_f_2)
    u_expected_2 = np.zeros(ndof)
    u_expected_2[free2] = u_f_expected_2
    K_sf_2 = K_global[np.ix_(fixed2, free2)]
    P_s_2 = P_global[fixed2]
    r_expected_2 = np.zeros(ndof)
    r_expected_2[fixed2] = K_sf_2 @ u_f_expected_2 - P_s_2
    u2, r2 = fcn(P_global, K_global, bc2, n_nodes)
    assert u2.shape == (ndof,)
    assert r2.shape == (ndof,)
    assert np.allclose(u2[fixed2], 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(r2[free2], 0.0, atol=1e-12, rtol=0.0)
    assert np.allclose(K_ff_2 @ u2[free2], P_f_2, atol=1e-10, rtol=1e-10)
    assert np.allclose(r2[fixed2], r_expected_2[fixed2], atol=1e-10, rtol=1e-10)
    residual2 = K_global @ u2 - P_global - r2
    assert np.allclose(residual2, 0.0, atol=1e-10, rtol=1e-10)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 1
    ndof = 6 * n_nodes
    K_global = np.zeros((ndof, ndof))
    P_global = np.zeros(ndof)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)