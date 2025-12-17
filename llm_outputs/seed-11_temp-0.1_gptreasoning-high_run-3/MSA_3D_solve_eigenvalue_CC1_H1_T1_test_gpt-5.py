def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 1
    diag_vals = np.array([2.0, 3.0, 5.0, 7.0, 11.0, 13.0])
    K_e_global = np.diag(diag_vals)
    K_g_global = -np.eye(6)
    boundary_conditions = {}
    lam, phi = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert phi.shape == (6 * n_nodes,)
    assert lam > 0
    assert np.isclose(lam, diag_vals.min(), rtol=1e-12, atol=1e-12)
    idx_min = int(np.argmin(diag_vals))
    assert np.argmax(np.abs(phi)) == idx_min
    others = np.delete(np.abs(phi), idx_min)
    if others.size > 0:
        ratio = others.max() / (abs(phi[idx_min]) + 1e-30)
        assert ratio < 1e-08

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 1
    K_e_global = np.zeros((6, 6))
    K_g_global = -np.eye(6)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 1
    K_e_global = np.zeros((6, 6))
    K_e_global[:2, :2] = np.array([[0.0, 1.0], [-1.0, 0.0]])
    K_g_global = -np.eye(6)
    boundary_conditions = {0: [False, False, True, True, True, True]}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 1
    diag_vals = -np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    K_e_global = np.diag(diag_vals)
    K_g_global = -np.eye(6)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """
    Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size.
    """
    n_nodes = 2
    size = 6 * n_nodes
    diag_vals = np.arange(1.0, size + 1.0)
    K_e_global = np.diag(diag_vals)
    K_g_global = -np.eye(size)
    bc_node0 = [False, False, False, True, True, True]
    bc_node1 = [False, False, False, False, False, False]
    boundary_conditions = {0: bc_node0, 1: bc_node1}
    lam_base, phi_base = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert phi_base.shape == (size,)
    assert lam_base > 0
    c = 3.5
    lam_scaled, phi_scaled = fcn(K_e_global, c * K_g_global, boundary_conditions, n_nodes)
    assert phi_scaled.shape == (size,)
    assert lam_scaled > 0
    assert np.isclose(lam_scaled, lam_base / c, rtol=1e-12, atol=1e-12)
    denom = np.dot(phi_base, phi_base)
    assert denom > 0
    alpha = np.dot(phi_scaled, phi_base) / denom
    rel_err = np.linalg.norm(phi_scaled - alpha * phi_base) / np.linalg.norm(phi_base)
    assert rel_err < 1e-08
    fixed_indices = []
    for node in range(n_nodes):
        bc = boundary_conditions.get(node, [False] * 6)
        for i in range(6):
            if bc[i]:
                fixed_indices.append(node * 6 + i)
    if fixed_indices:
        assert np.all(np.abs(phi_base[fixed_indices]) < 1e-12)
        assert np.all(np.abs(phi_scaled[fixed_indices]) < 1e-12)