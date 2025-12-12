def test_partition_dofs_correctness(fcn):
    """Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints."""
    boundary_conditions_empty = {}
    n_nodes = 3
    (fixed, free) = fcn(boundary_conditions_empty, n_nodes)
    assert len(fixed) == 0, 'No constraints should result in empty fixed array'
    assert len(free) == 6 * n_nodes, 'All DOFs should be free when no constraints'
    assert np.array_equal(free, np.arange(6 * n_nodes)), 'Free DOFs should be 0 to 17'
    n_nodes = 2
    boundary_conditions_full = {0: [True, True, True, True, True, True], 1: [True, True, True, True, True, True]}
    (fixed, free) = fcn(boundary_conditions_full, n_nodes)
    assert len(free) == 0, 'Full constraints should result in empty free array'
    assert len(fixed) == 6 * n_nodes, 'All DOFs should be fixed'
    assert np.array_equal(fixed, np.arange(6 * n_nodes)), 'Fixed DOFs should be 0 to 11'
    n_nodes = 2
    boundary_conditions_partial = {0: [True, True, True, False, False, False]}
    (fixed, free) = fcn(boundary_conditions_partial, n_nodes)
    expected_fixed = np.array([0, 1, 2])
    expected_free = np.array([3, 4, 5, 6, 7, 8, 9, 10, 11])
    assert np.array_equal(fixed, expected_fixed), 'Fixed DOFs should be translations at node 0'
    assert np.array_equal(free, expected_free), 'Free DOFs should be rotations at node 0 and all at node 1'
    n_nodes = 4
    boundary_conditions_nonconsec = {0: [True, False, True, False, True, False], 2: [False, True, False, True, False, True]}
    (fixed, free) = fcn(boundary_conditions_nonconsec, n_nodes)
    expected_fixed = np.array([0, 2, 4, 13, 15, 17])
    total_dofs = set(range(6 * n_nodes))
    expected_free = np.array(sorted(total_dofs - set(expected_fixed)))
    assert np.array_equal(fixed, expected_fixed), 'Fixed DOFs should match nonconsecutive pattern'
    assert np.array_equal(free, expected_free), 'Free DOFs should be complement of fixed'
    all_dofs = set(range(6 * n_nodes))
    assert set(fixed).union(set(free)) == all_dofs, 'Union of fixed and free should cover all DOFs'
    assert set(fixed).isdisjoint(set(free)), 'Fixed and free should be disjoint'
    n_nodes = 0
    boundary_conditions_zero = {}
    (fixed, free) = fcn(boundary_conditions_zero, n_nodes)
    assert len(fixed) == 0, 'Zero nodes should have no fixed DOFs'
    assert len(free) == 0, 'Zero nodes should have no free DOFs'
    n_nodes = 1
    boundary_conditions_single = {0: [True, False, True, False, True, False]}
    (fixed, free) = fcn(boundary_conditions_single, n_nodes)
    expected_fixed = np.array([0, 2, 4])
    expected_free = np.array([1, 3, 5])
    assert np.array_equal(fixed, expected_fixed), 'Single node fixed DOFs should be correct'
    assert np.array_equal(free, expected_free), 'Single node free DOFs should be correct'
    n_nodes = 3
    boundary_conditions_unsorted = {2: [True, True, False, False, False, False], 0: [False, False, True, True, False, False]}
    (fixed, free) = fcn(boundary_conditions_unsorted, n_nodes)
    assert np.all(fixed[:-1] <= fixed[1:]), 'Fixed array should be sorted'
    assert np.all(free[:-1] <= free[1:]), 'Free array should be sorted'
    assert len(fixed) == len(np.unique(fixed)), 'Fixed array should have unique values'
    assert len(free) == len(np.unique(free)), 'Free array should have unique values'