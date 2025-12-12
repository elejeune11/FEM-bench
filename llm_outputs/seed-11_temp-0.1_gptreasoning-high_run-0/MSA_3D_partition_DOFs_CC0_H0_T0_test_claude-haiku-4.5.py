def test_partition_dofs_correctness(fcn):
    """Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints."""
    (fixed, free) = fcn({}, 3)
    assert len(fixed) == 0
    assert len(free) == 18
    assert np.array_equal(free, np.arange(18))
    assert len(np.intersect1d(fixed, free)) == 0
    boundary_conditions = {0: [True, True, True, True, True, True]}
    (fixed, free) = fcn(boundary_conditions, 1)
    assert len(fixed) == 6
    assert len(free) == 0
    assert np.array_equal(fixed, np.arange(6))
    boundary_conditions = {0: [True, False, True, False, False, False]}
    (fixed, free) = fcn(boundary_conditions, 1)
    assert len(fixed) == 2
    assert len(free) == 4
    assert np.array_equal(fixed, np.array([0, 2]))
    assert np.array_equal(free, np.array([1, 3, 4, 5]))
    boundary_conditions = {0: [True, True, True, False, False, False], 2: [False, False, False, True, True, True]}
    (fixed, free) = fcn(boundary_conditions, 4)
    assert len(fixed) == 6
    assert len(free) == 18
    expected_fixed = np.array([0, 1, 2, 15, 16, 17])
    assert np.array_equal(fixed, expected_fixed)
    assert len(np.intersect1d(fixed, free)) == 0
    assert len(np.union1d(fixed, free)) == 24
    boundary_conditions = {0: [True, True, True, True, True, True], 1: [True, True, True, True, True, True], 2: [True, True, True, True, True, True]}
    (fixed, free) = fcn(boundary_conditions, 3)
    assert len(fixed) == 18
    assert len(free) == 0
    assert np.array_equal(fixed, np.arange(18))
    boundary_conditions = {1: [True, False, False, False, False, False], 3: [False, True, False, False, False, False], 5: [False, False, True, False, False, False]}
    (fixed, free) = fcn(boundary_conditions, 6)
    assert len(fixed) == 3
    assert len(free) == 33
    expected_fixed = np.array([6, 19, 32])
    assert np.array_equal(fixed, expected_fixed)
    assert len(np.union1d(fixed, free)) == 36
    (fixed, free) = fcn({}, 0)
    assert len(fixed) == 0
    assert len(free) == 0
    boundary_conditions = {0: [True, True, True, False, False, False], 99: [False, False, False, True, True, True]}
    (fixed, free) = fcn(boundary_conditions, 100)
    assert len(fixed) == 6
    assert len(free) == 594
    assert len(np.union1d(fixed, free)) == 600
    assert len(np.intersect1d(fixed, free)) == 0
    boundary_conditions = {2: [True, False, False, False, False, False]}
    (fixed, free) = fcn(boundary_conditions, 5)
    expected_fixed_dof = 6 * 2 + 0
    assert expected_fixed_dof in fixed
    assert len(fixed) == 1
    boundary_conditions = {0: [True, True, False, False, False, False], 1: [False, False, True, True, False, False]}
    (fixed, free) = fcn(boundary_conditions, 3)
    assert np.all(fixed[:-1] <= fixed[1:])
    assert np.all(free[:-1] <= free[1:])
    assert len(fixed) == len(np.unique(fixed))
    assert len(free) == len(np.unique(free))