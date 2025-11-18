def test_eigen_known_answer(fcn):
    """Verifies that eigenvalue_analysis produces the correct result in a simple,
    analytically solvable case. For example, with diagonal K_e and K_g = -I, the critical
    load factors reduce to the diagonal entries of K_e, so the function should
    return the smallest one and a mode aligned with the corresponding DOF."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_global = -np.eye(n_dofs)

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
    current_module = sys.modules[fcn.__module__]
    original_partition = getattr(current_module, 'partition_degrees_of_freedom', None)
    current_module.partition_degrees_of_freedom = partition_degrees_of_freedom
    try:
        (lambda_cr, mode) = fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
        expected_lambda = 7.0
        assert abs(lambda_cr - expected_lambda) < 1e-10
        assert len(mode) == n_dofs
    finally:
        if original_partition is not None:
            current_module.partition_degrees_of_freedom = original_partition

def test_eigen_singluar_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the
    reduced elastic block is singular/ill-conditioned."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.zeros((n_dofs, n_dofs))
    K_g_global = -np.eye(n_dofs)

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
    current_module = sys.modules[fcn.__module__]
    original_partition = getattr(current_module, 'partition_degrees_of_freedom', None)
    current_module.partition_degrees_of_freedom = partition_degrees_of_freedom
    try:
        with pytest.raises(ValueError):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        if original_partition is not None:
            current_module.partition_degrees_of_freedom = original_partition

def test_eigen_complex_eigenpairs_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when the generalized
    eigenproblem yields significantly complex pairs."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.eye(n_dofs)
    K_g_global = np.zeros((n_dofs, n_dofs))
    K_g_global[6, 7] = 1.0
    K_g_global[7, 8] = -1.0
    K_g_global = K_g_global - K_g_global.T

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
    current_module = sys.modules[fcn.__module__]
    original_partition = getattr(current_module, 'partition_degrees_of_freedom', None)
    current_module.partition_degrees_of_freedom = partition_degrees_of_freedom
    try:
        with pytest.raises(ValueError):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        if original_partition is not None:
            current_module.partition_degrees_of_freedom = original_partition

def test_eigen_no_positive_eigenvalues_detected(fcn):
    """Verify that eigenvalue_analysis raises ValueError when no positive
    eigenvalues are present."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.eye(n_dofs)
    K_g_global = np.eye(n_dofs)

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
    current_module = sys.modules[fcn.__module__]
    original_partition = getattr(current_module, 'partition_degrees_of_freedom', None)
    current_module.partition_degrees_of_freedom = partition_degrees_of_freedom
    try:
        with pytest.raises(ValueError):
            fcn(K_e_global, K_g_global, boundary_conditions, n_nodes)
    finally:
        if original_partition is not None:
            current_module.partition_degrees_of_freedom = original_partition

def test_eigen_invariance_to_reference_load_scaling(fcn):
    """Check that the computed critical load factor scales correctly with the
    reference geometric stiffness. Scaling K_g by a constant c should scale
    the reported eigenvalue by 1/c, while still returning valid global mode
    vectors of the correct size."""
    n_nodes = 2
    n_dofs = 6 * n_nodes
    K_e_global = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0])
    K_g_base = -np.eye(n_dofs)

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
    current_module = sys.modules[fcn.__module__]
    original_partition = getattr(current_module, 'partition_degrees_of_freedom', None)
    current_module.partition_degrees_of_freedom = partition_degrees_of_freedom
    try:
        (lambda1, mode1) = fcn(K_e_global, K_g_base, boundary_conditions, n_nodes)
        scale_factor = 2.0
        K_g_scaled = scale_factor * K_g_base
        (lambda2, mode2) = fcn(K_e_global, K_g_scaled, boundary_conditions, n_nodes)
        expected_lambda2 = lambda1 / scale_factor
        assert abs(lambda2 - expected_lambda2) < 1e-10
        assert len(mode1) == n_dofs
        assert len(mode2) == n_dofs
    finally:
        if original_partition is not None:
            current_module.partition_degrees_of_freedom = original_partition