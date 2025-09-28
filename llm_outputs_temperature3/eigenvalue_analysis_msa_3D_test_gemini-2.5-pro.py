def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    total_dofs = n_nodes * 6
    K_e_global = np.diag(np.arange(1, total_dofs + 1, dtype=float))
    K_g_global = -np.eye(total_dofs)
    constrained_dofs = list(range(6))
    boundary_conditions = type('BC', (), {'constrained_dofs': constrained_dofs})()
    expected_factor = 7.0
    (factor, shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(factor, expected_factor)
    assert shape.shape == (total_dofs,)
    assert np.allclose(shape[constrained_dofs], 0)
    assert not np.isclose(shape[6], 0)
    other_free_dofs = list(range(7, total_dofs))
    assert np.allclose(shape[other_free_dofs], 0)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    total_dofs = n_nodes * 6
    K_e_diag = np.ones(total_dofs, dtype=float)
    K_e_diag[6] = 0.0
    K_e_global = np.diag(K_e_diag)
    K_g_global = -np.eye(total_dofs)
    constrained_dofs = list(range(6))
    boundary_conditions = type('BC', (), {'constrained_dofs': constrained_dofs})()
    with pytest.raises(ValueError, match='ill-conditioned/singular'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    total_dofs = n_nodes * 6
    K_e_global = np.eye(total_dofs)
    K_e_global[6, 7] = 10.0
    K_e_global[7, 6] = -10.0
    K_g_global = -np.eye(total_dofs)
    constrained_dofs = list(range(6))
    boundary_conditions = type('BC', (), {'constrained_dofs': constrained_dofs})()
    with pytest.raises(ValueError, match='non-negligible complex parts'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    total_dofs = n_nodes * 6
    K_e_global = np.eye(total_dofs)
    K_g_global = np.eye(total_dofs)
    constrained_dofs = list(range(6))
    boundary_conditions = type('BC', (), {'constrained_dofs': constrained_dofs})()
    with pytest.raises(ValueError, match='no positive eigenvalue is found'):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    total_dofs = n_nodes * 6
    K_e_global = np.diag(np.arange(1, total_dofs + 1, dtype=float))
    K_g_global = -np.eye(total_dofs)
    constrained_dofs = list(range(6))
    boundary_conditions = type('BC', (), {'constrained_dofs': constrained_dofs})()
    (factor_1, vec_1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    c = 2.5
    K_g_scaled = c * K_g_global
    (factor_2, vec_2) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
    assert np.isclose(factor_2, factor_1 / c)
    assert vec_1.shape == (total_dofs,)
    assert vec_2.shape == (total_dofs,)
    norm_1 = np.linalg.norm(vec_1)
    norm_2 = np.linalg.norm(vec_2)
    assert not np.isclose(norm_1, 0)
    assert not np.isclose(norm_2, 0)
    vec_1_norm = vec_1 / norm_1
    vec_2_norm = vec_2 / norm_2
    dot_product = np.dot(vec_1_norm, vec_2_norm)
    assert np.isclose(abs(dot_product), 1.0)