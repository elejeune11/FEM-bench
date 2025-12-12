def test_partition_dofs_correctness(fcn):
    """Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
across representative cases including no constraints, full constraints,
partial constraints, and nonconsecutive node constraints."""
    n_nodes = 3
    boundary_conditions = {}
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 0
    assert len(free) == 6 * n_nodes
    assert np.array_equal(free, np.arange(6 * n_nodes))
    n_nodes = 2
    boundary_conditions = {0: [True, True, True, True, True, True], 1: [True, True, True, True, True, True]}
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 6 * n_nodes
    assert len(free) == 0
    assert np.array_equal(fixed, np.arange(6 * n_nodes))
    n_nodes = 3
    boundary_conditions = {0: [True, True, True, False, False, False]}
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 3
    assert len(free) == 6 * n_nodes - 3
    assert np.array_equal(fixed, np.array([0, 1, 2]))
    assert 0 not in free and 1 not in free and (2 not in free)
    assert np.all(np.isin(free, np.arange(6 * n_nodes)))
    n_nodes = 5
    boundary_conditions = {1: [True, False, True, False, True, False], 3: [False, True, False, True, False, True]}
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    expected_fixed = np.array([6, 8, 10, 19, 21, 23])
    expected_free_count = 6 * n_nodes - len(expected_fixed)
    assert len(fixed) == len(expected_fixed)
    assert len(free) == expected_free_count
    assert np.array_equal(fixed, expected_fixed)
    assert len(np.union1d(fixed, free)) == 6 * n_nodes
    assert len(np.intersect1d(fixed, free)) == 0
    n_nodes = 4
    boundary_conditions = {0: [True, True, True, True, True, True], 2: [True, False, False, False, False, False]}
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert 0 in fixed and 1 in fixed and (2 in fixed) and (3 in fixed) and (4 in fixed) and (5 in fixed)
    assert 12 in fixed
    assert len(fixed) == 7
    assert len(free) == 6 * n_nodes - 7
    assert np.all(fixed == np.sort(fixed))
    assert np.all(free == np.sort(free))
    n_nodes = 1
    boundary_conditions = {}
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 0
    assert len(free) == 6
    assert np.array_equal(free, np.array([0, 1, 2, 3, 4, 5]))
    n_nodes = 1
    boundary_conditions = {0: [True, True, True, True, True, True]}
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 6
    assert len(free) == 0
    assert np.array_equal(fixed, np.array([0, 1, 2, 3, 4, 5]))
    n_nodes = 0
    boundary_conditions = {}
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 0
    assert len(free) == 0
    n_nodes = 3
    boundary_conditions = {0: [True, False, True, False, True, False], 1: [False, True, False, True, False, True], 2: [True, False, True, False, True, False]}
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    expected_fixed_indices = [0, 2, 4, 7, 9, 11, 12, 14, 16]
    assert np.array_equal(fixed, np.array(expected_fixed_indices))
    assert len(free) == 6 * n_nodes - len(expected_fixed_indices)
    n_nodes = 6
    boundary_conditions = {1: [True, True, False, False, False, False], 3: [False, False, True, True, False, False], 5: [False, False, False, False, True, True]}
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    union = np.union1d(fixed, free)
    intersection = np.intersect1d(fixed, free)
    assert len(intersection) == 0
    assert np.array_equal(union, np.arange(6 * n_nodes))
    assert np.all(fixed == np.sort(fixed))
    assert np.all(free == np.sort(free))