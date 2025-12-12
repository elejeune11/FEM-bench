def test_partition_dofs_correctness(fcn):
    """
    Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints.
    """
    boundary_conditions = {}
    n_nodes = 3
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    expected_free = np.arange(18)
    expected_fixed = np.array([], dtype=int)
    assert np.array_equal(fixed, expected_fixed)
    assert np.array_equal(free, expected_free)
    boundary_conditions = {0: [True] * 6, 1: [True] * 6, 2: [True] * 6}
    n_nodes = 3
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    expected_fixed = np.arange(18)
    expected_free = np.array([], dtype=int)
    assert np.array_equal(fixed, expected_fixed)
    assert np.array_equal(free, expected_free)
    boundary_conditions = {0: [True, False, True, False, True, False], 2: [False, False, True, True, False, False]}
    n_nodes = 3
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    expected_fixed = np.array([0, 2, 4, 14, 15])
    expected_free = np.setdiff1d(np.arange(18), expected_fixed)
    assert np.array_equal(fixed, expected_fixed)
    assert np.array_equal(free, expected_free)
    boundary_conditions = {1: [True, True, False, False, False, False]}
    n_nodes = 4
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    expected_fixed = np.array([6, 7])
    expected_free = np.setdiff1d(np.arange(24), expected_fixed)
    assert np.array_equal(fixed, expected_fixed)
    assert np.array_equal(free, expected_free)
    boundary_conditions = {0: [True, False, False, False, False, True], 3: [False, True, False, False, False, False]}
    n_nodes = 5
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    expected_fixed = np.array([0, 5, 19])
    expected_free = np.setdiff1d(np.arange(30), expected_fixed)
    assert np.array_equal(fixed, expected_fixed)
    assert np.array_equal(free, expected_free)