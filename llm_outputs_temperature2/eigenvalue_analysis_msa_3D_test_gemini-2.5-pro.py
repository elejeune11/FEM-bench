def test_eigen_known_answer(fcn):
    """
    Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    k_diag = np.arange(10, 10 + n_dofs, dtype=float)
    k_diag[6] = 5.0
    K_e_global = np.diag(k_diag)
    K_g_global = -np.identity(n_dofs)

    class BC:
        pass
    boundary_conditions = BC()
    boundary_conditions.constrained_dofs = list(range(6))
    (lambda_cr, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    expected_lambda = 5.0
    assert np.isclose(lambda_cr, expected_lambda)
    assert mode_shape.shape == (n_dofs,)
    assert np.allclose(mode_shape[:6], 0.0)
    assert np.abs(mode_shape[6]) > 1e-09
    assert np.allclose(mode_shape[7:], 0.0)

def test_eigen_singluar_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.identity(n_dofs)
    K_e_global[6, 6] = 1e-17
    K_g_global = -np.identity(n_dofs)

    class BC:
        pass
    boundary_conditions = BC()
    boundary_conditions.constrained_dofs = list(range(6))
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.identity(n_dofs)
    K_e_global[6, 7] = 100.0
    K_e_global[7, 6] = -100.0
    K_g_global = -np.identity(n_dofs)

    class BC:
        pass
    boundary_conditions = BC()
    boundary_conditions.constrained_dofs = list(range(6))
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """
    Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present.
    """
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = -np.identity(n_dofs)
    K_g_global = -np.identity(n_dofs)

    class BC:
        pass
    boundary_conditions = BC()
    boundary_conditions.constrained_dofs = list(range(6))
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
    n_dofs = 6 * n_nodes
    k_diag = np.arange(10, 10 + n_dofs, dtype=float)
    k_diag[6] = 5.0
    K_e_global = np.diag(k_diag)
    K_g_global_base = -np.identity(n_dofs)

    class BC:
        pass
    boundary_conditions = BC()
    boundary_conditions.constrained_dofs = list(range(6))
    (lambda_base, mode_base) = fcn(K_e_global, K_g_global_base, boundary_conditions, n_nodes)
    scale_factor = 2.5
    K_g_global_scaled = scale_factor * K_g_global_base
    (lambda_scaled, mode_scaled) = fcn(K_e_global, K_g_global_scaled, boundary_conditions, n_nodes)
    assert np.isclose(lambda_scaled, lambda_base / scale_factor)
    assert mode_scaled.shape == mode_base.shape
    norm_base = np.linalg.norm(mode_base)
    norm_scaled = np.linalg.norm(mode_scaled)
    assert norm_base > 1e-09 and norm_scaled > 1e-09
    normalized_base = mode_base / norm_base
    normalized_scaled = mode_scaled / norm_scaled
    assert np.isclose(np.abs(np.dot(normalized_base, normalized_scaled)), 1.0)