def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    N = 6 * n_nodes
    d = np.array([7.0, 11.0, 13.0, 17.0, 19.0, 23.0, 2.0, 29.0, 31.0, 37.0, 41.0, 43.0])
    K_e_global = np.diag(d)
    K_g_global = -np.eye(N)
    boundary_conditions = {}
    lam, mode = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    idx_expected = int(np.argmin(d))
    assert np.isclose(lam, d[idx_expected], rtol=1e-10, atol=1e-12)
    assert mode.shape == (N,)
    idx_max = int(np.argmax(np.abs(mode)))
    assert idx_max == idx_expected
    assert np.abs(mode[idx_max]) > 1e-12
    others = np.delete(mode, idx_max)
    assert np.linalg.norm(others) <= 1e-08

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 1
    N = 6 * n_nodes
    K_e_global = np.zeros((N, N), dtype=float)
    K_g_global = -np.eye(N)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 1
    N = 6 * n_nodes
    K_e_global = np.zeros((N, N), dtype=float)
    K_e_global[:2, :2] = np.array([[0.0, -1.0], [1.0, 0.0]])
    K_g_global = -np.eye(N)
    bc_node0 = [False, False, True, True, True, True]
    boundary_conditions = {0: bc_node0}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 1
    N = 6 * n_nodes
    d = np.array([-5.0, 0.0, -3.0, -2.0, 0.0, -4.0])
    K_e_global = np.diag(d)
    K_g_global = -np.eye(N)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    N = 6 * n_nodes
    d = np.linspace(2.0, 2.0 + 2.0 * (N - 1), N)
    K_e_global = np.diag(d)
    bc_node0 = [False, True, False, True, False, False]
    bc_node1 = [False, False, False, False, False, True]
    boundary_conditions = {0: bc_node0, 1: bc_node1}

    def fixed_indices(n_nodes_local, bc):
        fixed = np.zeros(6 * n_nodes_local, dtype=bool)
        for node in range(n_nodes_local):
            if node in bc:
                arr = bc[node]
                for j in range(6):
                    if bool(arr[j]):
                        fixed[node * 6 + j] = True
        return np.where(fixed)[0]
    fixed_idx = fixed_indices(n_nodes, boundary_conditions)
    K_g1 = -np.eye(N)
    lam1, mode1 = fcn(K_e_global, K_g1, boundary_conditions, n_nodes)
    assert mode1.shape == (N,)
    if fixed_idx.size > 0:
        assert np.allclose(mode1[fixed_idx], 0.0, atol=1e-12)
    c = 2.5
    K_g2 = c * K_g1
    lam2, mode2 = fcn(K_e_global, K_g2, boundary_conditions, n_nodes)
    assert mode2.shape == (N,)
    if fixed_idx.size > 0:
        assert np.allclose(mode2[fixed_idx], 0.0, atol=1e-12)
    assert np.isclose(lam2, lam1 / c, rtol=1e-10, atol=1e-12)