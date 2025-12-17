def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """

    def fixed_free_from_bc(boundary_conditions, n_nodes):
        fixed = []
        free = []
        for i in range(n_nodes):
            bc = boundary_conditions.get(i, [False] * 6)
            bc = np.asarray(bc, dtype=bool)
            for j in range(6):
                idx = 6 * i + j
                if bc[j]:
                    fixed.append(idx)
                else:
                    free.append(idx)
        return (np.array(sorted(fixed), dtype=int), np.array(sorted(free), dtype=int))
    n_nodes = 2
    dof = 6 * n_nodes
    rng = np.random.default_rng(42)
    A = rng.normal(size=(dof, dof))
    K_global = A.T @ A + 10.0 * np.eye(dof)
    P_global = np.zeros(dof)
    P_global[6:] = rng.normal(size=6)
    boundary_conditions = {0: [True, True, True, True, True, True]}
    u, r = fcn(P_global, K_global, boundary_conditions, n_nodes)
    fixed, free = fixed_free_from_bc(boundary_conditions, n_nodes)
    assert u.shape == (dof,)
    assert r.shape == (dof,)
    assert np.allclose(u[fixed], 0.0, atol=1e-12)
    K_ff = K_global[np.ix_(free, free)]
    K_sf = K_global[np.ix_(fixed, free)]
    P_f = P_global[free]
    P_s = P_global[fixed]
    assert np.allclose(K_ff @ u[free], P_f, rtol=1e-09, atol=1e-09)
    r_expected = K_sf @ u[free] - P_s
    assert np.allclose(r[fixed], r_expected, rtol=1e-09, atol=1e-09)
    assert np.allclose(r[free], 0.0, atol=1e-12)
    residual = K_global @ u - P_global
    assert np.allclose(residual, r, rtol=1e-09, atol=1e-09)
    n_nodes = 3
    dof = 6 * n_nodes
    rng = np.random.default_rng(123)
    A = rng.normal(size=(dof, dof))
    K_global = A.T @ A + 5.0 * np.eye(dof)
    P_global = rng.normal(size=dof)
    boundary_conditions = {0: [True, True, True, False, False, False], 2: [False, True, False, True, False, True]}
    u, r = fcn(P_global, K_global, boundary_conditions, n_nodes)
    fixed, free = fixed_free_from_bc(boundary_conditions, n_nodes)
    assert u.shape == (dof,)
    assert r.shape == (dof,)
    assert np.allclose(u[fixed], 0.0, atol=1e-12)
    K_ff = K_global[np.ix_(free, free)]
    K_sf = K_global[np.ix_(fixed, free)]
    P_f = P_global[free]
    P_s = P_global[fixed]
    assert np.allclose(K_ff @ u[free], P_f, rtol=1e-09, atol=1e-09)
    r_expected = K_sf @ u[free] - P_s
    assert np.allclose(r[fixed], r_expected, rtol=1e-09, atol=1e-09)
    assert np.allclose(r[free], 0.0, atol=1e-12)
    residual = K_global @ u - P_global
    assert np.allclose(residual, r, rtol=1e-09, atol=1e-09)
    n_nodes = 1
    dof = 6 * n_nodes
    rng = np.random.default_rng(7)
    A = rng.normal(size=(dof, dof))
    K_global = A.T @ A + 2.0 * np.eye(dof)
    P_global = rng.normal(size=dof)
    boundary_conditions = {}
    u, r = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert np.allclose(r, 0.0, atol=1e-12)
    assert np.allclose(K_global @ u, P_global, rtol=1e-09, atol=1e-09)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 2
    dof = 6 * n_nodes
    bc_node0 = [False, False, True, True, True, True]
    bc_node1 = [True, True, True, True, True, True]
    boundary_conditions = {0: bc_node0, 1: bc_node1}
    K_global = np.zeros((dof, dof), dtype=float)
    K_global[0, 0] = 1.0
    K_global[1, 1] = 1e-20
    for i in range(2, dof):
        K_global[i, i] = 1.0
    P_global = np.ones(dof)
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)