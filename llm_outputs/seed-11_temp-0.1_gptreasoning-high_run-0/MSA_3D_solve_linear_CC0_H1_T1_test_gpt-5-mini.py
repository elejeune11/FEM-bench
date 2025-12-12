def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Verifies linear-solve on small 6-DOF-per-node systems: checks boundary handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium."""
    import numpy as np

    def _partition_indices(boundary_conditions, n_nodes):
        ndof = 6 * n_nodes
        fixed_mask = np.zeros(ndof, dtype=bool)
        for (node, vals) in boundary_conditions.items():
            vals = np.asarray(vals, dtype=bool)
            if vals.size != 6:
                raise ValueError('Boundary condition length must be 6 per node')
            for (j, v) in enumerate(vals):
                fixed_mask[6 * node + j] = bool(v)
        fixed = np.where(fixed_mask)[0]
        free = np.where(~fixed_mask)[0]
        return (fixed, free)
    rng = np.random.default_rng(1)
    n_nodes = 2
    ndof = 6 * n_nodes
    bc = {0: [True] * 6}
    A = rng.standard_normal((ndof, ndof))
    K_global = A.T @ A + np.eye(ndof) * 10.0
    P_global = np.zeros(ndof, dtype=float)
    P_global[6 + 0] = 100.0
    P_global[6 + 2] = -50.0
    P_global[6 + 3] = 10.0
    (fixed_idx, free_idx) = _partition_indices(bc, n_nodes)
    K_ff = K_global[np.ix_(free_idx, free_idx)]
    P_f = P_global[free_idx]
    u_f_expected = np.linalg.solve(K_ff, P_f)
    u_expected = np.zeros(ndof, dtype=float)
    u_expected[free_idx] = u_f_expected
    K_sf = K_global[np.ix_(fixed_idx, free_idx)]
    P_s = P_global[fixed_idx]
    r_fixed_expected = K_sf @ u_f_expected - P_s
    r_expected = np.zeros(ndof, dtype=float)
    r_expected[fixed_idx] = r_fixed_expected
    (u_out, r_out) = fcn(P_global.copy(), K_global.copy(), bc, n_nodes)
    assert isinstance(u_out, (list, tuple, np.ndarray, np.generic)) or hasattr(u_out, '__array__')
    assert isinstance(r_out, (list, tuple, np.ndarray, np.generic)) or hasattr(r_out, '__array__')
    u_out = np.asarray(u_out, dtype=float)
    r_out = np.asarray(r_out, dtype=float)
    assert u_out.shape == (ndof,)
    assert r_out.shape == (ndof,)
    assert np.allclose(u_out[free_idx], u_f_expected, rtol=1e-08, atol=1e-10)
    assert np.allclose(u_out[fixed_idx], 0.0, rtol=1e-12, atol=1e-12)
    assert np.allclose(K_ff @ u_out[free_idx], P_f, rtol=1e-08, atol=1e-10)
    assert np.allclose(r_out[fixed_idx], r_fixed_expected, rtol=1e-08, atol=1e-10)
    assert np.allclose(K_global @ u_out, P_global + r_out, rtol=1e-08, atol=1e-10)
    rng = np.random.default_rng(2)
    n_nodes = 3
    ndof = 6 * n_nodes
    bc = {0: [True, True, True, False, False, False], 1: [False, True, True, False, False, True]}
    A = rng.standard_normal((ndof, ndof))
    K_global = A.T @ A + np.eye(ndof) * 5.0
    P_global = rng.standard_normal(ndof) * 20.0
    (fixed_idx, free_idx) = _partition_indices(bc, n_nodes)
    K_ff = K_global[np.ix_(free_idx, free_idx)]
    P_f = P_global[free_idx]
    u_f_expected = np.linalg.solve(K_ff, P_f)
    u_expected = np.zeros(ndof, dtype=float)
    u_expected[free_idx] = u_f_expected
    K_sf = K_global[np.ix_(fixed_idx, free_idx)]
    P_s = P_global[fixed_idx]
    r_fixed_expected = K_sf @ u_f_expected - P_s
    r_expected = np.zeros(ndof, dtype=float)
    r_expected[fixed_idx] = r_fixed_expected
    (u_out, r_out) = fcn(P_global.copy(), K_global.copy(), bc, n_nodes)
    u_out = np.asarray(u_out, dtype=float)
    r_out = np.asarray(r_out, dtype=float)
    assert u_out.shape == (ndof,)
    assert r_out.shape == (ndof,)
    assert np.allclose(u_out[free_idx], u_f_expected, rtol=1e-08, atol=1e-10)
    assert np.allclose(u_out[fixed_idx], 0.0, rtol=1e-12, atol=1e-12)
    assert np.allclose(K_ff @ u_out[free_idx], P_f, rtol=1e-08, atol=1e-10)
    assert np.allclose(r_out[fixed_idx], r_fixed_expected, rtol=1e-08, atol=1e-10)
    assert np.allclose(K_global @ u_out, P_global + r_out, rtol=1e-08, atol=1e-10)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned."""
    import numpy as np
    import pytest
    n_nodes = 2
    ndof = 6 * n_nodes
    bc = {0: [True] * 6}
    K_global = np.zeros((ndof, ndof), dtype=float)
    K_global[:6, :6] = np.eye(6) * 1.0
    K_global[:6, 6:] = np.zeros((6, 6))
    K_global[6:, :6] = np.zeros((6, 6))
    K_global[6:, 6:] = np.zeros((6, 6))
    P_global = np.zeros(ndof, dtype=float)
    with pytest.raises(ValueError):
        fcn(P_global, K_global, bc, n_nodes)