def test_partition_dofs_correctness(fcn):
    """Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints."""
    (fixed, free) = fcn({}, 0)
    assert isinstance(fixed, np.ndarray) and isinstance(free, np.ndarray)
    assert fixed.size == 0 and free.size == 0
    assert len(set(fixed).intersection(set(free))) == 0
    assert set(fixed).union(set(free)) == set()
    N = 3
    (fixed, free) = fcn({}, N)
    assert fixed.size == 0
    assert np.array_equal(free, np.arange(6 * N))
    assert len(set(fixed) & set(free)) == 0
    assert set(fixed) | set(free) == set(range(6 * N))
    assert np.all(free[:-1] <= free[1:])
    N = 2
    bc = {i: [True] * 6 for i in range(N)}
    (fixed, free) = fcn(bc, N)
    assert np.array_equal(fixed, np.arange(6 * N))
    assert free.size == 0
    assert len(set(fixed) & set(free)) == 0
    assert set(fixed) | set(free) == set(range(6 * N))
    assert np.all(fixed[:-1] <= fixed[1:])
    N = 2
    bc = {0: [True, False, True, False, False, True]}
    (fixed, free) = fcn(bc, N)
    expected_fixed = np.array([0, 2, 5], dtype=int)
    expected_free = np.array(sorted(set(range(6 * N)) - set(expected_fixed)), dtype=int)
    assert np.array_equal(fixed, expected_fixed)
    assert np.array_equal(free, expected_free)
    assert len(set(fixed) & set(free)) == 0
    assert set(fixed) | set(free) == set(range(6 * N))
    N = 4
    bc = {1: [False, True, False, False, True, False], 3: [True, True, True, True, True, True]}
    (fixed, free) = fcn(bc, N)
    fixed_node1 = [6 * 1 + 1, 6 * 1 + 4]
    fixed_node3 = list(range(6 * 3, 6 * 3 + 6))
    expected_fixed = np.array(sorted(fixed_node1 + fixed_node3), dtype=int)
    expected_free = np.array(sorted(set(range(6 * N)) - set(expected_fixed)), dtype=int)
    assert np.array_equal(fixed, expected_fixed)
    assert np.array_equal(free, expected_free)
    assert len(set(fixed) & set(free)) == 0
    assert set(fixed) | set(free) == set(range(6 * N))
    assert np.all(fixed[:-1] <= fixed[1:])
    assert np.all(free[:-1] <= free[1:])