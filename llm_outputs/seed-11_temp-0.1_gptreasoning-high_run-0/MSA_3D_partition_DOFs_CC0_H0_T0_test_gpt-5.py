def test_partition_dofs_correctness(fcn):
    """
    Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints.
    """
    import numpy as np

    def check_case(bc, n_nodes):
        (fixed, free) = fcn(bc, n_nodes)
        assert isinstance(fixed, np.ndarray)
        assert isinstance(free, np.ndarray)
        assert np.issubdtype(fixed.dtype, np.integer)
        assert np.issubdtype(free.dtype, np.integer)
        total = 6 * n_nodes
        if fixed.size > 1:
            assert np.all(np.diff(fixed) > 0)
        if free.size > 1:
            assert np.all(np.diff(free) > 0)
        assert fixed.size == np.unique(fixed).size
        assert free.size == np.unique(free).size
        if fixed.size > 0:
            assert fixed.min() >= 0
            assert fixed.max() < total
        if free.size > 0:
            assert free.min() >= 0
            assert free.max() < total
        assert set(fixed.tolist()).isdisjoint(set(free.tolist()))
        merged = np.sort(np.concatenate([fixed, free]))
        assert merged.size == total
        assert np.array_equal(merged, np.arange(total, dtype=int))
        expected_fixed = []
        for node in range(n_nodes):
            mask = bc.get(node, [False] * 6)
            assert len(mask) == 6
            base = 6 * node
            for local in range(6):
                if bool(mask[local]):
                    expected_fixed.append(base + local)
        expected_fixed = np.array(sorted(expected_fixed), dtype=int)
        expected_free = np.array(sorted(set(range(total)) - set(expected_fixed.tolist())), dtype=int)
        assert np.array_equal(fixed, expected_fixed)
        assert np.array_equal(free, expected_free)
    check_case({}, 0)
    check_case({}, 3)
    check_case({0: [True] * 6, 1: [True] * 6}, 2)
    check_case({0: [True, False, False, False, True, False], 2: [True, True, True, False, False, False]}, 3)
    check_case({1: [False, True, False, False, False, True], 3: [True, False, True, False, False, False]}, 5)