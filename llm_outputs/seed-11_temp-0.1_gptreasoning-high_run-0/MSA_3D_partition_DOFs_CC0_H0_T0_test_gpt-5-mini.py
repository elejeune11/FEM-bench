def test_partition_dofs_correctness(fcn):
    """
    Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including:
    Verifies that:
    """

    def _as_int_list(arr):
        return [int(x) for x in list(arr)]
    n_nodes = 3
    (fixed, free) = fcn({}, n_nodes)
    fixed_list = _as_int_list(fixed)
    free_list = _as_int_list(free)
    total = 6 * n_nodes
    assert set(fixed_list).isdisjoint(set(free_list))
    assert set(fixed_list).union(set(free_list)) == set(range(total))
    assert len(fixed_list) == 0
    assert set(free_list) == set(range(total))
    assert free_list == sorted(free_list)
    assert fixed_list == sorted(fixed_list)
    assert len(set(free_list)) == len(free_list)
    assert all((0 <= x < total for x in free_list))
    n_nodes = 2
    bc_full = {i: [True, True, True, True, True, True] for i in range(n_nodes)}
    (fixed, free) = fcn(bc_full, n_nodes)
    fixed_list = _as_int_list(fixed)
    free_list = _as_int_list(free)
    total = 6 * n_nodes
    assert set(fixed_list) == set(range(total))
    assert len(free_list) == 0
    assert set(fixed_list).isdisjoint(set(free_list))
    assert set(fixed_list).union(set(free_list)) == set(range(total))
    assert fixed_list == sorted(fixed_list)
    assert free_list == sorted(free_list)
    assert len(set(fixed_list)) == len(fixed_list)
    assert all((0 <= x < total for x in fixed_list))
    n_nodes = 4
    bc_partial = {0: [True, False, False, False, False, False], 2: [False, True, False, True, False, False], 3: [False, False, True, False, True, False]}
    (fixed, free) = fcn(bc_partial, n_nodes)
    fixed_list = _as_int_list(fixed)
    free_list = _as_int_list(free)
    total = 6 * n_nodes
    expected_fixed = {0, 6 * 2 + 1, 6 * 2 + 3, 6 * 3 + 2, 6 * 3 + 4}
    assert set(fixed_list) == expected_fixed
    assert set(fixed_list).isdisjoint(set(free_list))
    assert set(fixed_list).union(set(free_list)) == set(range(total))
    assert fixed_list == sorted(fixed_list)
    assert free_list == sorted(free_list)
    assert len(set(fixed_list)) == len(fixed_list)
    assert len(set(free_list)) == len(free_list)
    assert all((0 <= x < total for x in fixed_list + free_list))
    n_nodes = 5
    bc_nonconsec = {0: [True, True, False, False, False, False], 4: [False, False, True, False, False, True]}
    (fixed, free) = fcn(bc_nonconsec, n_nodes)
    fixed_list = _as_int_list(fixed)
    free_list = _as_int_list(free)
    total = 6 * n_nodes
    expected_fixed = {0, 1, 6 * 4 + 2, 6 * 4 + 5}
    assert set(fixed_list) == expected_fixed
    assert set(fixed_list).isdisjoint(set(free_list))
    assert set(fixed_list).union(set(free_list)) == set(range(total))
    assert fixed_list == sorted(fixed_list)
    assert free_list == sorted(free_list)
    assert len(set(fixed_list)) + len(set(free_list)) == total
    assert all((0 <= x < total for x in fixed_list + free_list))