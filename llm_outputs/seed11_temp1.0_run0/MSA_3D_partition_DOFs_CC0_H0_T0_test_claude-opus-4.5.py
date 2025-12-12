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
    assert np.array_equal(fixed, expected_fixed), 'Fixed DOFs should be first 3 of node 0'
    assert np.array_equal(free, expected_free), 'Free DOFs should be remaining DOFs'
    n_nodes = 4
    boundary_conditions_noncons = {0: [True, False, True, False, True, False], 2: [False, True, False, True, False, True]}
    (fixed, free) = fcn(boundary_conditions_noncons, n_nodes)
    expected_fixed_noncons = np.array([0, 2, 4, 13, 15, 17])
    assert np.array_equal(fixed, expected_fixed_noncons), 'Nonconsecutive fixed DOFs incorrect'
    assert len(fixed) + len(free) == 6 * n_nodes, 'Total DOFs should equal 6*n_nodes'
    all_dofs = set(range(6 * n_nodes))
    assert set(fixed).isdisjoint(set(free)), 'Fixed and free should be disjoint'
    assert set(fixed).union(set(free)) == all_dofs, 'Fixed and free should cover all DOFs'
    n_nodes = 0
    boundary_conditions_zero = {}
    (fixed, free) = fcn(boundary_conditions_zero, n_nodes)
    assert len(fixed) == 0, 'Zero nodes should have no fixed DOFs'
    assert len(free) == 0, 'Zero nodes should have no free DOFs'
    n_nodes = 1
    boundary_conditions_single = {0: [True, False, True, False, True, False]}
    (fixed, free) = fcn(boundary_conditions_single, n_nodes)
    expected_fixed_single = np.array([0, 2, 4])
    expected_free_single = np.array([1, 3, 5])
    assert np.array_equal(fixed, expected_fixed_single), 'Single node fixed DOFs incorrect'
    assert np.array_equal(free, expected_free_single), 'Single node free DOFs incorrect'
    n_nodes = 3
    boundary_conditions_verify = {2: [True, True, True, True, True, True], 0: [True, True, True, True, True, True]}
    (fixed, free) = fcn(boundary_conditions_verify, n_nodes)
    assert np.all(np.diff(fixed) > 0), 'Fixed DOFs should be sorted'
    assert np.all(np.diff(free) > 0), 'Free DOFs should be sorted'
    assert len(fixed) == len(np.unique(fixed)), 'Fixed DOFs should be unique'
    assert len(free) == len(np.unique(free)), 'Free DOFs should be unique'