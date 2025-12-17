def test_partition_dofs_correctness(fcn):
    """
    Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints.
    """
    import numpy as np
    n = 3
    fixed, free = fcn({}, n)
    assert isinstance(fixed, np.ndarray)
    assert isinstance(free, np.ndarray)
    assert fixed.ndim == 1 and free.ndim == 1
    assert fixed.size == 0
    assert free.size == 6 * n
    assert np.array_equal(free, np.arange(6 * n, dtype=int))
    assert fixed.dtype.kind in ('i', 'u')
    assert free.dtype.kind in ('i', 'u')
    fixed0, free0 = fcn({}, 0)
    assert isinstance(fixed0, np.ndarray) and isinstance(free0, np.ndarray)
    assert fixed0.ndim == 1 and free0.ndim == 1
    assert fixed0.size == 0 and free0.size == 0
    n = 2
    bc = {0: [True] * 6, 1: [True] * 6}
    fixed, free = fcn(bc, n)
    assert np.array_equal(fixed, np.arange(6 * n, dtype=int))
    assert free.size == 0
    n = 2
    bc = {0: [True, False, False, False, False, True], 1: [False, True, True, True, False, False]}
    fixed_expected = np.array([0, 5, 7, 8, 9], dtype=int)
    fixed, free = fcn(bc, n)
    assert np.array_equal(fixed, fixed_expected)
    all_dofs = np.arange(6 * n, dtype=int)
    free_expected = np.setdiff1d(all_dofs, fixed_expected)
    assert np.array_equal(free, free_expected)
    assert np.intersect1d(fixed, free).size == 0
    assert np.array_equal(np.union1d(fixed, free), all_dofs)
    for arr in (fixed, free):
        if arr.size:
            assert np.all(np.diff(arr) > 0)
    n = 4
    bc = {0: [False, True, False, False, False, False], 3: [False, False, False, True, True, False]}
    fixed_expected = np.array([1, 18 + 3, 18 + 4], dtype=int)
    fixed_expected.sort()
    fixed, free = fcn(bc, n)
    assert np.array_equal(fixed, fixed_expected)
    all_dofs = np.arange(6 * n, dtype=int)
    free_expected = np.setdiff1d(all_dofs, fixed_expected)
    assert np.array_equal(free, free_expected)
    assert np.all(np.isin(np.arange(6 * 1, 6 * 2, dtype=int), free))
    assert np.all(np.isin(np.arange(6 * 2, 6 * 3, dtype=int), free))
    assert np.array_equal(np.union1d(fixed, free), all_dofs)