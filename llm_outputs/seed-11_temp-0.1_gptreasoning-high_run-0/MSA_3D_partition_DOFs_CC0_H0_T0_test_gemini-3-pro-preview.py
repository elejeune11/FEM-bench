def test_partition_dofs_correctness(fcn):
    """
    Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints.
    """
    n_nodes_1 = 1
    bc_1 = {0: [True, False, False, False, False, True]}
    (fixed_1, free_1) = fcn(bc_1, n_nodes_1)
    expected_fixed_1 = np.array([0, 5])
    expected_free_1 = np.array([1, 2, 3, 4])
    np.testing.assert_array_equal(fixed_1, expected_fixed_1)
    np.testing.assert_array_equal(free_1, expected_free_1)
    n_nodes_2 = 3
    bc_2 = {1: [True] * 6}
    (fixed_2, free_2) = fcn(bc_2, n_nodes_2)
    expected_fixed_2 = np.arange(6, 12)
    expected_free_2 = np.concatenate((np.arange(0, 6), np.arange(12, 18)))
    np.testing.assert_array_equal(fixed_2, expected_fixed_2)
    np.testing.assert_array_equal(free_2, expected_free_2)
    n_nodes_3 = 2
    bc_3 = {}
    (fixed_3, free_3) = fcn(bc_3, n_nodes_3)
    assert fixed_3.size == 0
    np.testing.assert_array_equal(free_3, np.arange(12))
    n_nodes_4 = 1
    bc_4 = {0: [True] * 6}
    (fixed_4, free_4) = fcn(bc_4, n_nodes_4)
    np.testing.assert_array_equal(fixed_4, np.arange(6))
    assert free_4.size == 0
    n_nodes_5 = 0
    bc_5 = {}
    (fixed_5, free_5) = fcn(bc_5, n_nodes_5)
    assert fixed_5.size == 0
    assert free_5.size == 0
    n_nodes_6 = 2
    bc_6 = {0: [False, True, False, True, False, True], 1: [True, False, True, False, True, False]}
    (fixed_6, free_6) = fcn(bc_6, n_nodes_6)
    all_dofs = np.concatenate((fixed_6, free_6))
    all_dofs.sort()
    np.testing.assert_array_equal(all_dofs, np.arange(12))
    intersection = np.intersect1d(fixed_6, free_6)
    assert intersection.size == 0
    assert np.all(fixed_6[:-1] <= fixed_6[1:])
    assert np.all(free_6[:-1] <= free_6[1:])