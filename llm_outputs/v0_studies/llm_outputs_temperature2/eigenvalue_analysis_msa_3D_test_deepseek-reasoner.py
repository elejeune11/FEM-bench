def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case."""
    import numpy as np
    import pytest
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(dof)
    boundary_conditions = {'fixed_dofs': list(range(6))}
    (load_factor, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert load_factor == pytest.approx(7.0, rel=1e-10)
    assert mode_shape.shape == (dof,)
    assert np.argmax(np.abs(mode_shape)) == 6

def test_eigen_singular_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    import numpy as np
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.zeros((dof, dof))
    K_g_global = -np.eye(dof)
    boundary_conditions = {'fixed_dofs': list(range(6))}
    try:
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
        assert False, 'Expected ValueError for singular matrix'
    except ValueError as e:
        assert 'ill-conditioned' in str(e).lower() or 'singular' in str(e).lower()

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    import numpy as np
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.eye(dof)
    K_g_global = np.random.rand(dof, dof)
    K_g_global = K_g_global + K_g_global.T
    K_g_global[0, 1] = 10.0
    K_g_global[1, 0] = -10.0
    boundary_conditions = {'fixed_dofs': list(range(6))}
    try:
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    except ValueError as e:
        assert 'complex' in str(e).lower()

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    import numpy as np
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = -np.eye(dof)
    K_g_global = np.eye(dof)
    boundary_conditions = {'fixed_dofs': list(range(6))}
    try:
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
        assert False, 'Expected ValueError for no positive eigenvalues'
    except ValueError as e:
        assert 'positive' in str(e).lower()

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness."""
    import numpy as np
    import pytest
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.diag([10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0, 110.0, 120.0])
    K_g_global = -np.eye(dof)
    boundary_conditions = {'fixed_dofs': list(range(6))}
    (load_factor1, mode_shape1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    scaling_factor = 2.5
    (load_factor2, mode_shape2) = fcn(K_e_global, scaling_factor * K_g_global, boundary_conditions, n_nodes)
    assert load_factor2 == pytest.approx(load_factor1 / scaling_factor, rel=1e-10)
    mode1_norm = mode_shape1 / np.linalg.norm(mode_shape1)
    mode2_norm = mode_shape2 / np.linalg.norm(mode_shape2)
    dot_product = np.abs(np.dot(mode1_norm, mode2_norm))
    assert dot_product == pytest.approx(1.0, rel=1e-10)