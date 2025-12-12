def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    import numpy as np
    n_nodes = 1
    size = 6 * n_nodes
    diag = np.array([10.0, 2.0, 5.0, 7.0, 9.0, 3.0], dtype=float)
    K_e_global = np.diag(diag)
    K_g_global = -np.eye(size, dtype=float)
    boundary_conditions = {}
    (elastic_critical_load_factor, deformed_shape_vector) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    expected = float(diag.min())
    assert abs(elastic_critical_load_factor - expected) < 1e-10
    assert deformed_shape_vector.shape == (size,)
    min_idx = int(np.argmin(diag))
    assert abs(deformed_shape_vector[min_idx]) > 0.0
    for i in range(size):
        if i != min_idx:
            assert abs(deformed_shape_vector[i]) < 1e-08

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    import numpy as np
    n_nodes = 1
    size = 6 * n_nodes
    diag = np.array([1e-18, 1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
    K_e_global = np.diag(diag)
    K_g_global = -np.eye(size, dtype=float)
    boundary_conditions = {}
    try:
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    except ValueError:
        return
    else:
        assert False, 'Expected ValueError due to ill-conditioned/singular reduced elastic stiffness'

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    import numpy as np
    n_nodes = 1
    size = 6 * n_nodes
    K_e_global = np.zeros((size, size), dtype=float)
    K_e_global[0:2, 0:2] = np.array([[0.0, -1.0], [1.0, 0.0]])
    K_e_global[2:, 2:] = np.eye(size - 2, dtype=float) * 2.0
    K_g_global = -np.eye(size, dtype=float)
    boundary_conditions = {}
    try:
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    except ValueError:
        return
    else:
        assert False, 'Expected ValueError due to significantly complex eigenpairs'

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    import numpy as np
    n_nodes = 1
    size = 6 * n_nodes
    K_e_global = -np.eye(size, dtype=float)
    K_g_global = -np.eye(size, dtype=float)
    boundary_conditions = {}
    try:
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    except ValueError:
        return
    else:
        assert False, 'Expected ValueError because there are no positive eigenvalues'

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    import numpy as np
    n_nodes = 1
    size = 6 * n_nodes
    diag = np.array([10.0, 6.0, 8.0, 14.0, 20.0, 12.0], dtype=float)
    K_e_global = np.diag(diag)
    K_g_global = -np.eye(size, dtype=float)
    boundary_conditions = {}
    (lam1, phi1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    c = 3.0
    (lam2, phi2) = fcn(K_e_global, c * K_g_global, boundary_conditions, n_nodes)
    assert phi1.shape == (size,)
    assert phi2.shape == (size,)
    assert abs(lam2 - lam1 / c) < 1e-10
    idx1 = int(np.argmax(np.abs(phi1)))
    idx2 = int(np.argmax(np.abs(phi2)))
    assert idx1 == idx2
    assert abs(phi1[idx1]) > 0.0
    assert abs(phi2[idx2]) > 0.0