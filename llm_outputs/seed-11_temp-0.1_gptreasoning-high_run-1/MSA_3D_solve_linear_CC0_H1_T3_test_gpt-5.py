def test_linear_solve_arbitrary_solvable_cases(fcn):
    """
    Verifies `linear_solve` against small, solvable 6-DOF-per-node systems that
    mimic cantilever-style setups. Checks boundary-condition handling,
    free-DOF equilibrium (K_ff u_f = P_f), reactions at fixed DOFs, and global equilibrium.
    """

    def build_two_node_spring_K(k):
        ndofs = 12
        K = np.zeros((ndofs, ndofs), dtype=float)
        for j in range(6):
            i_s = j
            i_f = 6 + j
            K[i_s, i_s] += k[j]
            K[i_f, i_f] += k[j]
            K[i_s, i_f] += -k[j]
            K[i_f, i_s] += -k[j]
        return K
    n_nodes = 2
    ndofs = 6 * n_nodes
    bc = {0: np.array([True, True, True, True, True, True], dtype=bool)}
    k = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=float)
    K = build_two_node_spring_K(k)
    P = np.zeros(ndofs, dtype=float)
    P_f = np.array([2.0, -3.0, 4.0, -5.0, 6.0, -7.0], dtype=float)
    P[6:] = P_f
    u, r = fcn(P, K, bc, n_nodes)
    fixed_mask = np.zeros(ndofs, dtype=bool)
    fixed_mask[:6] = True
    free_mask = ~fixed_mask
    assert np.allclose(u[fixed_mask], 0.0)
    assert np.allclose(r[free_mask], 0.0)
    K_ff = K[np.ix_(free_mask, free_mask)]
    assert np.allclose(K_ff @ u[free_mask], P[free_mask])
    K_sf = K[np.ix_(fixed_mask, free_mask)]
    expected_r_fixed = K_sf @ u[free_mask] - P[fixed_mask]
    assert np.allclose(r[fixed_mask], expected_r_fixed)
    assert np.allclose(K @ u - P - r, 0.0)
    expected_u_f = P_f / k
    assert np.allclose(u[free_mask], expected_u_f)
    P2 = np.zeros(ndofs, dtype=float)
    P2[:6] = np.array([1.0, -1.5, 2.0, -2.5, 3.0, -3.5], dtype=float)
    P2[6:] = np.array([-4.0, 0.5, -0.25, 1.25, -2.0, 0.75], dtype=float)
    u2, r2 = fcn(P2, K, bc, n_nodes)
    assert np.allclose(u2[fixed_mask], 0.0)
    assert np.allclose(r2[free_mask], 0.0)
    assert np.allclose(K_ff @ u2[free_mask], P2[free_mask])
    expected_r_fixed2 = K_sf @ u2[free_mask] - P2[fixed_mask]
    assert np.allclose(r2[fixed_mask], expected_r_fixed2)
    assert np.allclose(K @ u2 - P2 - r2, 0.0)

def test_linear_solve_raises_on_ill_conditioned_Kff(fcn):
    """
    Ensures ValueError is raised when the free–free stiffness submatrix (K_ff) is ill-conditioned.
    """
    n_nodes = 1
    ndofs = 6 * n_nodes
    K_global = np.zeros((ndofs, ndofs), dtype=float)
    P_global = np.zeros(ndofs, dtype=float)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(P_global, K_global, boundary_conditions, n_nodes)