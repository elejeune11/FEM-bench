def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.diag([1.0, 3.0, 2.0, 5.0, 4.0, 6.0, 8.0, 7.0, 9.0, 10.0, 12.0, 11.0])
    K_g_global = -np.eye(dof)

    class BoundaryConditions:

        def __init__(self, constrained_dofs):
            self.constrained_dofs = constrained_dofs
    boundary_conditions = BoundaryConditions(constrained_dofs=[])
    (critical_load, mode_shape) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    expected_load = 1.0
    expected_mode = np.zeros(dof)
    expected_mode[0] = 1.0
    assert np.isclose(critical_load, expected_load)
    assert np.allclose(mode_shape, expected_mode)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.zeros((dof, dof))
    K_g_global = -np.eye(dof)

    class BoundaryConditions:

        def __init__(self, constrained_dofs):
            self.constrained_dofs = constrained_dofs
    boundary_conditions = BoundaryConditions(constrained_dofs=[])
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.eye(dof)
    K_g_global = np.zeros((dof, dof))
    K_g_global[0, 1] = 1.0
    K_g_global[1, 0] = -1.0

    class BoundaryConditions:

        def __init__(self, constrained_dofs):
            self.constrained_dofs = constrained_dofs
    boundary_conditions = BoundaryConditions(constrained_dofs=[])
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = -np.eye(dof)
    K_g_global = -np.eye(dof)

    class BoundaryConditions:

        def __init__(self, constrained_dofs):
            self.constrained_dofs = constrained_dofs
    boundary_conditions = BoundaryConditions(constrained_dofs=[])
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    dof = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(dof)

    class BoundaryConditions:

        def __init__(self, constrained_dofs):
            self.constrained_dofs = constrained_dofs
    boundary_conditions = BoundaryConditions(constrained_dofs=[])
    (critical_load_1, mode_shape_1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    scaling_factor = 2.0
    (critical_load_2, mode_shape_2) = fcn(K_e_global, scaling_factor * K_g_global, boundary_conditions, n_nodes)
    assert np.isclose(critical_load_2, critical_load_1 / scaling_factor)
    assert mode_shape_1.shape == mode_shape_2.shape
    assert np.allclose(np.abs(mode_shape_1), np.abs(mode_shape_2))