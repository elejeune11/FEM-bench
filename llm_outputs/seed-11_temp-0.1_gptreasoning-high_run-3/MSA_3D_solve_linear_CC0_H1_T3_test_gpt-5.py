def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    import numpy as np

    def build_masks(n_nodes, boundary_conditions):
        fixed = np.zeros(6 * n_nodes, dtype=bool)
        for node, mask in boundary_conditions.items():
            m = np.array(mask, dtype=bool)
            fixed[node * 6:node * 6 + 6] = m
        free = ~fixed
        return (fixed, free)
    rng = np.random.default_rng(42)
    n_nodes = 2
    N = 6 * n_nodes
    A = rng.standard_normal((N, N))
    K_global = A.T @ A + 10.0 * np.eye(N)
    P_global = rng.standard_normal(N)
    boundary_conditions = {0: [True, True, True, True, True, True]}
    u, r = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert u.shape == (N,)
    assert r.shape == (N,)
    fixed, free = build_masks(n_nodes, boundary_conditions)
    assert np.allclose(u[fixed], 0.0, atol=1e-12)
    K_ff = K_global[np.ix_(free, free)]
    P_f = P_global[free]
    u_f_expected = np.linalg.solve(K_ff, P_f)
    assert np.allclose(u[free], u_f_expected, rtol=1e-10, atol=1e-12)
    K_sf = K_global[np.ix_(fixed, free)]
    P_s = P_global[fixed]
    r_s_expected = K_sf @ u_f_expected - P_s
    assert np.allclose(r[free], 0.0, atol=1e-12)
    assert np.allclose(r[fixed], r_s_expected, rtol=1e-10, atol=1e-12)
    assert np.allclose(K_ff @ u[free], P_global[free], rtol=1e-10, atol=1e-12)
    residual = K_global @ u - P_global - r
    assert np.allclose(residual, 0.0, atol=1e-10)
    n_nodes = 3
    N = 6 * n_nodes
    A = rng.standard_normal((N, N))
    K_global = A.T @ A + 15.0 * np.eye(N)
    P_global = rng.standard_normal(N)
    boundary_conditions = {0: [True, True, True, True, True, True], 1: [True, False, True, False, True, False]}
    u, r = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert u.shape == (N,)
    assert r.shape == (N,)
    fixed, free = build_masks(n_nodes, boundary_conditions)
    assert np.allclose(u[fixed], 0.0, atol=1e-12)
    K_ff = K_global[np.ix_(free, free)]
    P_f = P_global[free]
    u_f_expected = np.linalg.solve(K_ff, P_f)
    assert np.allclose(u[free], u_f_expected, rtol=1e-10, atol=1e-12)
    K_sf = K_global[np.ix_(fixed, free)]
    P_s = P_global[fixed]
    r_s_expected = K_sf @ u_f_expected - P_s
    assert np.allclose(r[free], 0.0, atol=1e-12)
    assert np.allclose(r[fixed], r_s_expected, rtol=1e-10, atol=1e-12)
    assert np.allclose(K_ff @ u[free], P_global[free], rtol=1e-10, atol=1e-12)
    residual = K_global @ u - P_global - r
    assert np.allclose(residual, 0.0, atol=1e-10)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    import numpy as np
    import pytest
    n_nodes = 1
    N = 6 * n_nodes
    diag_entries = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1e-20])
    K_global = np.diag(diag_entries)
    P_global = np.arange(1, N + 1, dtype=float)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)