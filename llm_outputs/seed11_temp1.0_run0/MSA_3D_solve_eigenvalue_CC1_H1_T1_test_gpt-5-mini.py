def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    import numpy as np
    import pytest
    n_nodes = 1
    dof = 6 * n_nodes
    diag_vals = np.array([2.0, 3.0, 5.0, 10.0, 20.0, 30.0])
    K_e = np.diag(diag_vals)
    K_g = -np.eye(dof)
    boundary_conditions = {}
    (lam, mode) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    assert isinstance(lam, float)
    assert np.isclose(lam, diag_vals.min(), atol=1e-08)
    assert mode.shape == (dof,)
    idx_min = int(np.argmin(diag_vals))
    abs_mode = np.abs(mode)
    assert abs_mode[idx_min] > 1e-08
    for j in range(dof):
        if j != idx_min:
            assert abs_mode[j] / abs_mode[idx_min] < 1e-08

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    import numpy as np
    import pytest
    n_nodes = 1
    dof = 6 * n_nodes
    K_e = np.zeros((dof, dof))
    K_g = -np.eye(dof)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    import numpy as np
    import pytest
    n_nodes = 1
    dof = 6 * n_nodes
    K_e = np.eye(dof)
    K_g = np.zeros((dof, dof))
    K_g[0:2, 0:2] = np.array([[0.0, 1.0], [-1.0, 0.0]])
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    import numpy as np
    import pytest
    n_nodes = 1
    dof = 6 * n_nodes
    K_e = np.diag(np.array([-1.0, -2.0, -3.0, -4.0, -5.0, -6.0]))
    K_g = -np.eye(dof)
    boundary_conditions = {}
    with pytest.raises(ValueError):
        fcn(K_e, K_g, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    import numpy as np
    import pytest
    n_nodes = 1
    dof = 6 * n_nodes
    K_e = np.diag(np.array([4.0, 6.0, 8.0, 10.0, 12.0, 14.0]))
    K_g = -np.eye(dof)
    boundary_conditions = {}
    (lam1, v1) = fcn(K_e, K_g, boundary_conditions, n_nodes)
    c = 2.5
    (lam2, v2) = fcn(K_e, c * K_g, boundary_conditions, n_nodes)
    assert np.isclose(lam2, lam1 / c, rtol=1e-08, atol=1e-10)
    assert v1.shape == (dof,)
    assert v2.shape == (dof,)
    norm1 = np.linalg.norm(v1)
    norm2 = np.linalg.norm(v2)
    assert norm1 > 0 and norm2 > 0
    v1n = v1 / norm1
    v2n = v2 / norm2
    diff = np.linalg.norm(v1n - v2n)
    diff_flip = np.linalg.norm(v1n + v2n)
    assert min(diff, diff_flip) < 1e-06