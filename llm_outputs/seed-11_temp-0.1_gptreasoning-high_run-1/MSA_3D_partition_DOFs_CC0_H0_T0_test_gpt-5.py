def test_partition_dofs_correctness(fcn):
    """
    Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints.
    """
    import numpy as np

    def is_sorted_unique(arr):
        if arr.size <= 1:
            return True
        d = np.diff(arr)
        return np.all(d > 0)
    nA = 3
    bcA = {}
    fixed_exp_A = np.array([], dtype=int)
    free_exp_A = np.arange(6 * nA, dtype=int)
    fixed_A, free_A = fcn(bcA, nA)
    assert np.array_equal(fixed_A, fixed_exp_A)
    assert np.array_equal(free_A, free_exp_A)
    assert is_sorted_unique(fixed_A) and is_sorted_unique(free_A)
    assert len(np.intersect1d(fixed_A, free_A)) == 0
    assert np.array_equal(np.union1d(fixed_A, free_A), np.arange(6 * nA))
    assert len(fixed_A) + len(free_A) == 6 * nA
    assert np.issubdtype(fixed_A.dtype, np.integer)
    assert np.issubdtype(free_A.dtype, np.integer)
    nB = 2
    bcB = {0: [True] * 6, 1: [True] * 6}
    fixed_exp_B = np.arange(6 * nB, dtype=int)
    free_exp_B = np.array([], dtype=int)
    fixed_B, free_B = fcn(bcB, nB)
    assert np.array_equal(fixed_B, fixed_exp_B)
    assert np.array_equal(free_B, free_exp_B)
    assert is_sorted_unique(fixed_B) and is_sorted_unique(free_B)
    assert len(np.intersect1d(fixed_B, free_B)) == 0
    assert np.array_equal(np.union1d(fixed_B, free_B), np.arange(6 * nB))
    assert len(fixed_B) + len(free_B) == 6 * nB
    assert np.issubdtype(fixed_B.dtype, np.integer)
    assert np.issubdtype(free_B.dtype, np.integer)
    nC = 3
    bcC = {1: [True, False, False, False, False, True], 2: [False, True, True, True, False, False]}
    fixed_exp_C = np.array([6, 11, 13, 14, 15], dtype=int)
    free_exp_C = np.setdiff1d(np.arange(6 * nC, dtype=int), fixed_exp_C)
    fixed_C, free_C = fcn(bcC, nC)
    assert np.array_equal(fixed_C, fixed_exp_C)
    assert np.array_equal(free_C, free_exp_C)
    assert is_sorted_unique(fixed_C) and is_sorted_unique(free_C)
    assert len(np.intersect1d(fixed_C, free_C)) == 0
    assert np.array_equal(np.union1d(fixed_C, free_C), np.arange(6 * nC))
    assert len(fixed_C) + len(free_C) == 6 * nC
    assert np.issubdtype(fixed_C.dtype, np.integer)
    assert np.issubdtype(free_C.dtype, np.integer)
    nD = 5
    bcD = {0: [True, True, True, False, False, False], 3: [True, False, False, False, True, False], 4: [False, False, False, False, False, False]}
    fixed_exp_D = np.array([0, 1, 2, 18, 22], dtype=int)
    free_exp_D = np.setdiff1d(np.arange(6 * nD, dtype=int), fixed_exp_D)
    fixed_D, free_D = fcn(bcD, nD)
    assert np.array_equal(fixed_D, fixed_exp_D)
    assert np.array_equal(free_D, free_exp_D)
    assert is_sorted_unique(fixed_D) and is_sorted_unique(free_D)
    assert len(np.intersect1d(fixed_D, free_D)) == 0
    assert np.array_equal(np.union1d(fixed_D, free_D), np.arange(6 * nD))
    assert len(fixed_D) + len(free_D) == 6 * nD
    assert np.issubdtype(fixed_D.dtype, np.integer)
    assert np.issubdtype(free_D.dtype, np.integer)