def test_partition_dofs_correctness(fcn):
    """
    Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
    across representative cases including no constraints, full constraints,
    partial constraints, and nonconsecutive node constraints.
    """
    import numpy as np

    def assert_partition(fixed, free, n_nodes, expected_fixed=None, expected_free=None):
        fixed = np.asarray(fixed)
        free = np.asarray(free)
        if expected_fixed is not None:
            assert np.array_equal(fixed, expected_fixed)
        if expected_free is not None:
            assert np.array_equal(free, expected_free)
        assert fixed.size == np.unique(fixed).size
        assert free.size == np.unique(free).size
        if fixed.size > 0:
            assert np.all(np.diff(fixed) > 0)
        if free.size > 0:
            assert np.all(np.diff(free) > 0)
        assert np.intersect1d(fixed, free).size == 0
        assert fixed.size + free.size == 6 * n_nodes
        assert set(fixed.tolist() + free.tolist()) == set(range(6 * n_nodes))
    n_zero = 0
    bc_zero = {}
    fixed_z, free_z = fcn(bc_zero, n_zero)
    expected_fixed_z = np.array([], dtype=int)
    expected_free_z = np.array([], dtype=int)
    assert_partition(fixed_z, free_z, n_zero, expected_fixed=expected_fixed_z, expected_free=expected_free_z)
    n0 = 3
    bc0 = {}
    fixed0, free0 = fcn(bc0, n0)
    expected_fixed0 = np.array([], dtype=int)
    expected_free0 = np.arange(6 * n0)
    assert_partition(fixed0, free0, n0, expected_fixed=expected_fixed0, expected_free=expected_free0)
    n1 = 2
    bc1 = {i: [True] * 6 for i in range(n1)}
    fixed1, free1 = fcn(bc1, n1)
    expected_fixed1 = np.arange(6 * n1)
    expected_free1 = np.array([], dtype=int)
    assert_partition(fixed1, free1, n1, expected_fixed=expected_fixed1, expected_free=expected_free1)
    n2 = 4
    bc2 = {1: [True, False, True, False, False, False], 3: [False, False, False, True, True, False]}
    fixed2, free2 = fcn(bc2, n2)
    expected_fixed2 = np.array([6, 8, 21, 22], dtype=int)
    expected_free2 = np.setdiff1d(np.arange(6 * n2), expected_fixed2)
    assert_partition(fixed2, free2, n2, expected_fixed=expected_fixed2, expected_free=expected_free2)
    n3 = 5
    bc3 = {0: [True, False, False, True, False, True], 4: [False, True, True, False, False, False]}
    fixed3, free3 = fcn(bc3, n3)
    expected_fixed3 = np.array([0, 3, 5, 25, 26], dtype=int)
    expected_free3 = np.setdiff1d(np.arange(6 * n3), expected_fixed3)
    assert_partition(fixed3, free3, n3, expected_fixed=expected_fixed3, expected_free=expected_free3)