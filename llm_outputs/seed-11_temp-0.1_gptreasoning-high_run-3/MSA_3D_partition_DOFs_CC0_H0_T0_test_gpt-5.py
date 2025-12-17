def test_partition_dofs_correctness(fcn):
    """
    Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints.
    """
    import numpy as np

    def assert_partition_props(fixed, free, n_nodes):
        all_dofs = np.arange(6 * n_nodes, dtype=int)
        assert isinstance(fixed, np.ndarray)
        assert isinstance(free, np.ndarray)
        assert fixed.dtype.kind in 'iu'
        assert free.dtype.kind in 'iu'
        assert fixed.ndim == 1
        assert free.ndim == 1
        assert np.array_equal(fixed, np.sort(fixed))
        assert np.array_equal(free, np.sort(free))
        assert np.unique(fixed).size == fixed.size
        assert np.unique(free).size == free.size
        assert np.intersect1d(fixed, free).size == 0
        assert np.array_equal(np.union1d(fixed, free), all_dofs)
    N = 3
    fixed, free = fcn({}, N)
    assert fixed.size == 0
    assert np.array_equal(free, np.arange(6 * N))
    assert_partition_props(fixed, free, N)
    N = 3
    bc_full = {i: [True] * 6 for i in range(N)}
    fixed, free = fcn(bc_full, N)
    assert np.array_equal(fixed, np.arange(6 * N))
    assert free.size == 0
    assert_partition_props(fixed, free, N)
    N = 3
    bc_partial = {0: [True, False, True, False, False, True], 1: [False, True, False, True, False, False]}
    fixed, free = fcn(bc_partial, N)
    expected_fixed = np.array([0, 2, 5, 7, 9])
    expected_free = np.array([1, 3, 4, 6, 8, 10, 11, 12, 13, 14, 15, 16, 17])
    assert np.array_equal(fixed, expected_fixed)
    assert np.array_equal(free, expected_free)
    assert_partition_props(fixed, free, N)
    N = 5
    bc_nonconsec = {0: [True, False, False, True, False, False], 2: [False, False, True, False, True, True], 4: [False, True, False, False, False, False]}
    fixed, free = fcn(bc_nonconsec, N)
    expected_fixed = np.array([0, 3, 14, 16, 17, 25])
    expected_free = np.setdiff1d(np.arange(6 * N), expected_fixed)
    assert np.array_equal(fixed, expected_fixed)
    assert np.array_equal(free, expected_free)
    assert_partition_props(fixed, free, N)