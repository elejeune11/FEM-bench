def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case."""
    import numpy as np
    from unittest.mock import Mock
    n_nodes = 2
    dof = 6 * n_nodes
    K_e = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g = -np.eye(dof)
    bc = Mock()
    bc.constrained_dofs = list(range(6))
    (load_factor, mode_shape) = fcn(K_e, K_g, bc, n_nodes)
    assert np.isclose(load_factor, 7.0)
    assert mode_shape.shape == (dof,)
    assert np.argmax(np.abs(mode_shape)) == 6

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    import numpy as np
    from unittest.mock import Mock
    n_nodes = 2
    dof = 6 * n_nodes
    K_e = np.zeros((dof, dof))
    K_g = -np.eye(dof)
    bc = Mock()
    bc.constrained_dofs = [0, 1]
    try:
        fcn(K_e, K_g, bc, n_nodes)
        assert False, 'Expected ValueError for singular matrix'
    except ValueError as e:
        assert 'ill-conditioned' in str(e).lower() or 'singular' in str(e).lower()

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    import numpy as np
    from unittest.mock import Mock
    n_nodes = 2
    dof = 6 * n_nodes
    K_e = np.eye(dof)
    K_g = np.random.rand(dof, dof)
    K_g = K_g + K_g.T
    K_e[0, 1] = 10.0
    K_e[1, 0] = -10.0
    bc = Mock()
    bc.constrained_dofs = list(range(6))
    try:
        fcn(K_e, K_g, bc, n_nodes)
        assert False, 'Expected ValueError for complex eigenpairs'
    except ValueError as e:
        assert 'complex' in str(e).lower()

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    import numpy as np
    from unittest.mock import Mock
    n_nodes = 2
    dof = 6 * n_nodes
    K_e = -np.eye(dof)
    K_g = -np.eye(dof)
    bc = Mock()
    bc.constrained_dofs = list(range(6))
    try:
        fcn(K_e, K_g, bc, n_nodes)
        assert False, 'Expected ValueError for no positive eigenvalues'
    except ValueError as e:
        assert 'positive' in str(e).lower()

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness."""
    import numpy as np
    from unittest.mock import Mock
    n_nodes = 2
    dof = 6 * n_nodes
    K_e = np.diag(range(1, dof + 1))
    K_g_base = -np.eye(dof)
    bc = Mock()
    bc.constrained_dofs = list(range(6))
    (load_factor1, mode_shape1) = fcn(K_e, K_g_base, bc, n_nodes)
    scale_factor = 2.5
    (load_factor2, mode_shape2) = fcn(K_e, scale_factor * K_g_base, bc, n_nodes)
    assert np.isclose(load_factor2, load_factor1 / scale_factor)
    norm1 = np.linalg.norm(mode_shape1)
    norm2 = np.linalg.norm(mode_shape2)
    if norm1 > 0 and norm2 > 0:
        cos_similarity = np.abs(np.dot(mode_shape1 / norm1, mode_shape2 / norm2))
        assert np.isclose(cos_similarity, 1.0, atol=1e-10)