def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Verifies linear_solve against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    import numpy as np
    n_nodes = 1
    K_global = np.diag(np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=float))
    u_target = np.array([0.1, -0.2, 0.3, -0.01, 0.02, -0.03], dtype=float)
    P_global = K_global.dot(u_target)
    boundary_conditions = {}
    (u, r) = fcn(P_global.copy(), K_global.copy(), boundary_conditions, n_nodes)
    assert u.shape == (6 * n_nodes,)
    assert r.shape == (6 * n_nodes,)
    assert np.allclose(u, u_target, atol=1e-09, rtol=1e-09)
    assert np.allclose(K_global.dot(u), P_global, atol=1e-09, rtol=1e-09)
    assert np.allclose(r, np.zeros(6), atol=1e-12)
    n_nodes = 2
    K_ss = np.diag(np.array([1000.0, 1000.0, 1000.0, 500.0, 500.0, 500.0], dtype=float))
    K_sf = (np.arange(36).reshape(6, 6).astype(float) + 1.0) * 0.1
    K_fs = K_sf.T
    K_ff = np.diag(np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=float))
    K_global = np.zeros((12, 12), dtype=float)
    K_global[0:6, 0:6] = K_ss
    K_global[0:6, 6:12] = K_sf
    K_global[6:12, 0:6] = K_fs
    K_global[6:12, 6:12] = K_ff
    u_free_target = np.array([0.01, 0.02, 0.03, 0.001, 0.002, 0.003], dtype=float)
    P_fixed = np.zeros(6, dtype=float)
    P_free = K_ff.dot(u_free_target)
    P_global = np.zeros(12, dtype=float)
    P_global[0:6] = P_fixed
    P_global[6:12] = P_free
    boundary_conditions = {0: np.ones(6, dtype=bool)}
    (u, r) = fcn(P_global.copy(), K_global.copy(), boundary_conditions, n_nodes)
    assert u.shape == (12,)
    assert r.shape == (12,)
    assert np.allclose(u[0:6], np.zeros(6), atol=1e-12)
    assert np.allclose(u[6:12], u_free_target, atol=1e-09, rtol=1e-09)
    assert np.allclose(K_ff.dot(u[6:12]), P_free, atol=1e-09, rtol=1e-09)
    expected_reactions_fixed = K_sf.dot(u_free_target) - P_fixed
    assert np.allclose(r[0:6], expected_reactions_fixed, atol=1e-09, rtol=1e-09)
    assert np.allclose(r[6:12], np.zeros(6), atol=1e-12)
    lhs = K_global.dot(u)
    rhs = P_global + r
    assert np.allclose(lhs, rhs, atol=1e-09, rtol=1e-09)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """Ensures ValueError is raised when the free–free stiffness submatrix (K_ff)
    is ill-conditioned (cond(K_ff) >= 1e16).
    """
    import numpy as np
    import pytest
    n_nodes = 1
    K_global = np.diag(np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1e-20], dtype=float))
    P_global = np.zeros(6, dtype=float)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)