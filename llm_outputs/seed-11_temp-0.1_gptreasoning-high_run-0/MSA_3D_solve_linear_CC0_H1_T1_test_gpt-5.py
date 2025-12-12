def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """
    import numpy as np

    def get_indices(bc, n_nodes):
        mask = np.zeros((n_nodes, 6), dtype=bool)
        for (node, arr) in bc.items():
            a = np.asarray(arr, dtype=bool)
            assert a.shape == (6,)
            mask[node, :] = a
        fixed = np.flatnonzero(mask.ravel())
        all_idx = np.arange(6 * n_nodes)
        free = np.setdiff1d(all_idx, fixed, assume_unique=False)
        return (fixed, free)
    n_nodes = 2
    n = 6 * n_nodes
    B = np.eye(n) * 2.0 + np.tril(np.ones((n, n)), -1) * 0.1
    K = B.T @ B
    bc = {0: np.array([True, True, True, True, True, True])}
    P = np.zeros(n)
    P[0:6] = np.array([5.0, -3.0, 2.0, 1.0, -4.0, 6.0], dtype=float)
    P[6:12] = np.array([1.2, -0.5, 3.0, 0.7, -1.1, 2.2], dtype=float)
    (fixed, free) = get_indices(bc, n_nodes)
    K_ff = K[np.ix_(free, free)]
    K_sf = K[np.ix_(fixed, free)]
    P_f = P[free].copy()
    P_s = P[fixed].copy()
    u_f_expected = np.linalg.solve(K_ff, P_f)
    r_s_expected = K_sf @ u_f_expected - P_s
    u_expected = np.zeros(n)
    u_expected[free] = u_f_expected
    r_expected = np.zeros(n)
    r_expected[fixed] = r_s_expected
    (u, r) = fcn(P, K, bc, n_nodes)
    assert u.shape == (n,)
    assert r.shape == (n,)
    assert np.allclose(u, u_expected, atol=1e-10, rtol=1e-10)
    assert np.allclose(r, r_expected, atol=1e-10, rtol=1e-10)
    assert np.allclose(K_ff @ u_f_expected, P_f, atol=1e-10, rtol=1e-10)
    residual = K @ u - P - r
    assert np.allclose(residual, np.zeros_like(P), atol=1e-10, rtol=1e-10)
    assert np.allclose(u[fixed], 0.0)
    assert np.allclose(r[free], 0.0)
    n_nodes = 3
    n = 6 * n_nodes
    B = np.eye(n) * 1.7 + np.tril(np.ones((n, n)), -1) * 0.07
    K = B.T @ B + 0.1 * np.eye(n)
    bc = {0: np.array([True, True, True, False, False, False]), 2: np.array([False, False, False, True, True, True])}
    P = np.arange(n, dtype=float) - 4.5
    (fixed, free) = get_indices(bc, n_nodes)
    K_ff = K[np.ix_(free, free)]
    K_sf = K[np.ix_(fixed, free)]
    P_f = P[free].copy()
    P_s = P[fixed].copy()
    u_f_expected = np.linalg.solve(K_ff, P_f)
    r_s_expected = K_sf @ u_f_expected - P_s
    u_expected = np.zeros(n)
    u_expected[free] = u_f_expected
    r_expected = np.zeros(n)
    r_expected[fixed] = r_s_expected
    (u, r) = fcn(P, K, bc, n_nodes)
    assert u.shape == (n,)
    assert r.shape == (n,)
    assert np.allclose(u, u_expected, atol=1e-10, rtol=1e-10)
    assert np.allclose(r, r_expected, atol=1e-10, rtol=1e-10)
    assert np.allclose(K_ff @ u_f_expected, P_f, atol=1e-10, rtol=1e-10)
    residual = K @ u - P - r
    assert np.allclose(residual, np.zeros_like(P), atol=1e-10, rtol=1e-10)
    assert np.allclose(u[fixed], 0.0)
    assert np.allclose(r[free], 0.0)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    import numpy as np
    import pytest
    n_nodes = 1
    n = 6
    K = np.zeros((n, n), dtype=float)
    bc = {0: np.array([True, False, False, False, False, False], dtype=bool)}
    P = np.zeros(n, dtype=float)
    with pytest.raises(ValueError):
        fcn(P, K, bc, n_nodes)