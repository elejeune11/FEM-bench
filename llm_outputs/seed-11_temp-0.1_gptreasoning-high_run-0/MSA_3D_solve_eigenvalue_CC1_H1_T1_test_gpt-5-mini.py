def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    import numpy as np
    K_e_diag = np.array([2.0, 3.0, 1.5, 4.0, 5.0, 6.0], dtype=float)
    K_e_global = np.diag(K_e_diag)
    K_g_global = -np.eye(6, dtype=float)
    boundary_conditions = {}
    n_nodes = 1
    (elastic_critical_load_factor, deformed_shape_vector) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(elastic_critical_load_factor, K_e_diag.min(), rtol=1e-12, atol=1e-12)
    deformed_shape_vector = np.asarray(deformed_shape_vector)
    assert deformed_shape_vector.shape == (6,)
    nonzero_indices = np.where(np.abs(deformed_shape_vector) > 1e-08)[0]
    assert nonzero_indices.size == 1
    assert int(nonzero_indices[0]) == int(K_e_diag.argmin())

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    import numpy as np
    import pytest
    K_e_global = np.zeros((6, 6), dtype=float)
    K_g_global = -np.eye(6, dtype=float)
    boundary_conditions = {}
    n_nodes = 1
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    import numpy as np
    import pytest
    R = np.array([[0.0, -1.0], [1.0, 0.0]], dtype=float)
    rest_diag = np.array([2.0, 3.0, 4.0, 5.0], dtype=float)
    K_e_global = np.zeros((6, 6), dtype=float)
    K_e_global[0:2, 0:2] = R
    for (i, val) in enumerate(rest_diag, start=2):
        K_e_global[i, i] = val
    K_g_global = -np.eye(6, dtype=float)
    boundary_conditions = {}
    n_nodes = 1
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    import numpy as np
    import pytest
    K_e_diag = -np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=float)
    K_e_global = np.diag(K_e_diag)
    K_g_global = -np.eye(6, dtype=float)
    boundary_conditions = {}
    n_nodes = 1
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    import numpy as np
    K_e_diag = np.array([4.0, 2.0, 3.0, 5.0, 6.0, 7.0], dtype=float)
    K_e_global = np.diag(K_e_diag)
    K_g_global = -np.eye(6, dtype=float)
    boundary_conditions = {}
    n_nodes = 1
    (elastic_critical_load_factor_1, deformed_shape_vector_1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    c = 3.5
    (elastic_critical_load_factor_2, deformed_shape_vector_2) = fcn(K_e_global, c * K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(elastic_critical_load_factor_2, elastic_critical_load_factor_1 / c, rtol=1e-12, atol=1e-12)
    v1 = np.asarray(deformed_shape_vector_1, dtype=float)
    v2 = np.asarray(deformed_shape_vector_2, dtype=float)
    assert v1.shape == v2.shape == (6,)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    assert norm1 > 0 and norm2 > 0
    v1n = v1 / norm1
    v2n = v2 / norm2
    assert np.allclose(np.abs(v1n), np.abs(v2n), rtol=1e-08, atol=1e-10)