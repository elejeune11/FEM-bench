def test_partition_dofs_correctness(fcn):
    """
    Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints.
    """
    boundary_conditions = {}
    n_nodes = 5
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 0
    assert len(free) == 6 * n_nodes
    assert np.all(free == np.arange(6 * n_nodes))
    boundary_conditions = {i: [True] * 6 for i in range(n_nodes)}
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(free) == 0
    assert len(fixed) == 6 * n_nodes
    assert np.all(fixed == np.arange(6 * n_nodes))
    boundary_conditions = {0: [True, False, True, False, True, False], 2: [False, True, False, True, False, True]}
    n_nodes = 5
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    expected_fixed = np.array([0, 2, 4, 6 * 2 + 1, 6 * 2 + 3, 6 * 2 + 5])
    expected_free = np.setdiff1d(np.arange(6 * n_nodes), expected_fixed)
    assert np.all(fixed == expected_fixed)
    assert np.all(free == expected_free)
    boundary_conditions = {0: [True] * 6, 2: [True] * 6, 4: [True] * 6}
    n_nodes = 5
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    expected_fixed = np.array([0, 1, 2, 3, 4, 5, 6 * 2 + 0, 6 * 2 + 1, 6 * 2 + 2, 6 * 2 + 3, 6 * 2 + 4, 6 * 2 + 5, 6 * 4 + 0, 6 * 4 + 1, 6 * 4 + 2, 6 * 4 + 3, 6 * 4 + 4, 6 * 4 + 5])
    expected_free = np.setdiff1d(np.arange(6 * n_nodes), expected_fixed)
    assert np.all(fixed == expected_fixed)
    assert np.all(free == expected_free)