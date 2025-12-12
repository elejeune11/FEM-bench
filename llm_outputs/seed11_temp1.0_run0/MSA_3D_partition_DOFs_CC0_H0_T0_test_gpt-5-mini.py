def test_partition_dofs_correctness(fcn):
    """
    Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints.
    """
    import numpy as np

    def check(boundary_conditions, n_nodes, expected_fixed_set):
        (fixed, free) = fcn(boundary_conditions, n_nodes)
        fixed = np.asarray(fixed, dtype=int)
        free = np.asarray(free, dtype=int)
        total_dofs = 6 * n_nodes
        assert fixed.ndim == 1
        assert free.ndim == 1
        assert np.array_equal(fixed, np.sort(fixed))
        assert np.array_equal(free, np.sort(free))
        assert fixed.size == np.unique(fixed).size
        assert free.size == np.unique(free).size
        assert np.intersect1d(fixed, free).size == 0
        union = np.union1d(fixed, free)
        assert union.size == total_dofs
        assert np.array_equal(union, np.arange(total_dofs, dtype=int))
        expected = np.sort(np.asarray(list(expected_fixed_set), dtype=int))
        assert np.array_equal(fixed, expected)
    check({}, 3, expected_fixed_set=[])
    bc_full = {i: [True] * 6 for i in range(2)}
    check(bc_full, 2, expected_fixed_set=list(range(12)))
    bc_partial = {0: [True] * 6, 2: [True, False, True, False, False, True]}
    expected_fixed_partial = list(range(6)) + [6 * 2 + 0, 6 * 2 + 2, 6 * 2 + 5]
    check(bc_partial, 4, expected_fixed_set=expected_fixed_partial)
    import numpy as _np
    bc_nonconsec = {1: _np.array([False, True, False, False, True, False]), 4: (True, False, False, True, False, False)}
    expected_fixed_nonconsec = [6 * 1 + 1, 6 * 1 + 4, 6 * 4 + 0, 6 * 4 + 3]
    check(bc_nonconsec, 5, expected_fixed_set=expected_fixed_nonconsec)
    check({}, 0, expected_fixed_set=[])