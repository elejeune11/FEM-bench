def test_partition_dofs_correctness(fcn):
    """
    Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints.
    """
    import numpy as np

    def check_case(boundary_conditions, n_nodes, expected_fixed):
        fixed, free = fcn(boundary_conditions, n_nodes)
        assert isinstance(fixed, np.ndarray)
        assert isinstance(free, np.ndarray)
        assert fixed.ndim == 1
        assert free.ndim == 1
        assert fixed.dtype.kind in 'iu'
        assert free.dtype.kind in 'iu'
        assert np.all(fixed[:-1] < fixed[1:]) or fixed.size <= 1
        assert np.all(free[:-1] < free[1:]) or free.size <= 1
        assert fixed.size == np.unique(fixed).size
        assert free.size == np.unique(free).size
        total = 6 * n_nodes
        all_idx = np.arange(total, dtype=int)
        assert np.intersect1d(fixed, free).size == 0
        assert np.array_equal(np.sort(np.concatenate([fixed, free])), all_idx)
        assert free.size == total - fixed.size
        if expected_fixed is not None:
            expected_fixed = np.asarray(expected_fixed, dtype=int)
            assert np.array_equal(fixed, expected_fixed)
    N = 3
    bc = {}
    expected_fixed = np.array([], dtype=int)
    check_case(bc, N, expected_fixed)
    N = 0
    bc = {}
    expected_fixed = np.array([], dtype=int)
    check_case(bc, N, expected_fixed)
    N = 2
    bc = {0: [True] * 6, 1: [True] * 6}
    expected_fixed = np.arange(6 * N, dtype=int)
    check_case(bc, N, expected_fixed)
    N = 3
    bc = {0: [True, False, True, False, False, False], 1: [False, False, False, True, True, False]}
    expected_fixed = np.array([0, 2, 9, 10], dtype=int)
    check_case(bc, N, expected_fixed)
    N = 5
    bc = {0: [False, False, False, False, False, True], 3: [True, True, False, False, False, False]}
    expected_fixed = np.array([5, 18, 19], dtype=int)
    check_case(bc, N, expected_fixed)