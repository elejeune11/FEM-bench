def test_linear_solve_arbitrary_solvable_cases(fcn):
    """Verifies linear_solve on a small 2-node cantilever-like system:
      and global equilibrium K_global @ u == P_global + r.
    """
    import numpy as np
    n_nodes = 2
    dof_per_node = 6
    N = n_nodes * dof_per_node
    boundary_conditions = {0: np.array([True] * dof_per_node, dtype=bool)}
    K_ff = 10.0 * np.eye(dof_per_node)
    K_sf = np.arange(1, dof_per_node * dof_per_node + 1, dtype=float).reshape(dof_per_node, dof_per_node)
    K_fs = K_sf.T
    K_ss = 1.0 * np.eye(dof_per_node)
    K_global = np.block([[K_ss, K_sf], [K_fs, K_ff]])
    P_s = np.zeros(dof_per_node, dtype=float)
    P_f = np.arange(1.0, dof_per_node + 1.0)
    P_global = np.concatenate([P_s, P_f])
    (u, r) = fcn(P_global, K_global, boundary_conditions, n_nodes)
    assert u.shape == (N,)
    assert r.shape == (N,)
    assert np.allclose(u[:dof_per_node], 0.0)
    u_free = u[dof_per_node:]
    expected_u_free = np.linalg.solve(K_ff, P_f)
    assert np.allclose(u_free, expected_u_free)
    assert np.allclose(K_ff @ u_free, P_f)
    expected_r_fixed = K_sf @ u_free - P_s
    assert np.allclose(r[:dof_per_node], expected_r_fixed)
    assert np.allclose(r[dof_per_node:], 0.0)
    assert np.allclose(K_global @ u, P_global + r)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """Ensures ValueError is raised when the free–free stiffness submatrix K_ff is ill-conditioned."""
    import numpy as np
    import pytest
    n_nodes = 1
    dof_per_node = 6
    N = n_nodes * dof_per_node
    boundary_conditions = {}
    K_global = np.zeros((N, N), dtype=float)
    P_global = np.zeros(N, dtype=float)
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)