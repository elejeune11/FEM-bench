def test_partition_dofs_correctness(fcn):
    """
    Test that MSA_3D_partition_DOFs_CC0_H0_T0 correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints.
    """
    boundary_conditions: Dict[int, np.ndarray] = {}
    n_nodes = 2
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert np.array_equal(free, np.arange(12))
    assert len(fixed) == 0
    boundary_conditions = {i: np.ones(6, dtype=bool) for i in range(n_nodes)}
    n_nodes = 2
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert np.array_equal(fixed, np.arange(12))
    assert len(free) == 0
    boundary_conditions = {0: np.array([True, True, False, False, False, False])}
    n_nodes = 2
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert np.array_equal(fixed, np.array([0, 1]))
    assert np.array_equal(free, np.array([2, 3, 4, 5, 6, 7, 8, 9, 10, 11]))
    boundary_conditions = {0: np.array([True, False, False, False, False, False]), 2: np.array([False, False, True, False, False, False])}
    n_nodes = 4
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert np.array_equal(fixed, np.array([0, 8]))
    assert np.array_equal(free, np.array([1, 2, 3, 4, 5, 6, 7, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23]))