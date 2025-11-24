def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(n_dof)

    class BoundaryConditions:

        def __init__(self):
            self.constrained_dofs = list(range(6))
    boundary_conditions = BoundaryConditions()

    def partition_degrees_of_freedom(bc, n_nodes):
        all_dofs = set(range(6 * n_nodes))
        constrained = set(bc.constrained_dofs)
        free = sorted(all_dofs - constrained)
        return (free, sorted(constrained))
    import sys
    if 'partition_degrees_of_freedom' not in sys.modules:
        import types
        module = types.ModuleType('partition_degrees_of_freedom')
        module.partition_degrees_of_freedom = partition_degrees_of_freedom
        sys.modules['partition_degrees_of_freedom'] = module
    (lambda_cr, mode) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    assert abs(lambda_cr - 7.0) < 1e-10
    assert len(mode) == n_dof
    assert np.all(mode[:6] == 0)

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_e_global = np.zeros((n_dof, n_dof))
    K_g_global = -np.eye(n_dof)

    class BoundaryConditions:

        def __init__(self):
            self.constrained_dofs = list(range(6))
    boundary_conditions = BoundaryConditions()
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_e_global = np.eye(n_dof)
    K_g_global = np.eye(n_dof)
    K_g_global[6, 7] = 0.5
    K_g_global[7, 6] = -0.5

    class BoundaryConditions:

        def __init__(self):
            self.constrained_dofs = list(range(6))
    boundary_conditions = BoundaryConditions()
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_e_global = -np.eye(n_dof)
    K_g_global = -np.eye(n_dof)

    class BoundaryConditions:

        def __init__(self):
            self.constrained_dofs = list(range(6))
    boundary_conditions = BoundaryConditions()
    with pytest.raises(ValueError):
        fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    n_dof = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(n_dof)

    class BoundaryConditions:

        def __init__(self):
            self.constrained_dofs = list(range(6))
    boundary_conditions = BoundaryConditions()
    (lambda_1, mode_1) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    scale_factor = 2.0
    K_g_scaled = scale_factor * K_g_global
    (lambda_2, mode_2) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
    assert abs(lambda_2 - lambda_1 / scale_factor) < 1e-10
    assert len(mode_1) == len(mode_2) == n_dof
    assert np.all(mode_1[:6] == 0)
    assert np.all(mode_2[:6] == 0)