def test_partition_dofs_correctness(fcn):
    """Test that partition_degrees_of_freedom correctly separates fixed and free DOFs
across representative cases including no constraints, full constraints,
partial constraints, and nonconsecutive node constraints."""
    n_nodes = 3
    bcs = {}
    total_dofs = 6 * n_nodes
    expected_fixed = np.array([], dtype=int)
    expected_free = np.arange(total_dofs, dtype=int)
    (fixed, free) = fcn(boundary_conditions=bcs, n_nodes=n_nodes)
    np.testing.assert_array_equal(fixed, expected_fixed)
    np.testing.assert_array_equal(free, expected_free)
    assert isinstance(fixed, np.ndarray)
    assert isinstance(free, np.ndarray)
    assert len(np.intersect1d(fixed, free)) == 0
    assert len(fixed) + len(free) == total_dofs
    n_nodes = 2
    bcs = {0: [True] * 6, 1: [True] * 6}
    total_dofs = 6 * n_nodes
    expected_fixed = np.arange(total_dofs, dtype=int)
    expected_free = np.array([], dtype=int)
    (fixed, free) = fcn(boundary_conditions=bcs, n_nodes=n_nodes)
    np.testing.assert_array_equal(fixed, expected_fixed)
    np.testing.assert_array_equal(free, expected_free)
    n_nodes = 2
    bcs = {0: [True] * 6}
    expected_fixed = np.arange(6, dtype=int)
    expected_free = np.arange(6, 12, dtype=int)
    (fixed, free) = fcn(boundary_conditions=bcs, n_nodes=n_nodes)
    np.testing.assert_array_equal(fixed, expected_fixed)
    np.testing.assert_array_equal(free, expected_free)
    n_nodes = 1
    bcs = {0: [True, False, True, False, True, False]}
    expected_fixed = np.array([0, 2, 4], dtype=int)
    expected_free = np.array([1, 3, 5], dtype=int)
    (fixed, free) = fcn(boundary_conditions=bcs, n_nodes=n_nodes)
    np.testing.assert_array_equal(fixed, expected_fixed)
    np.testing.assert_array_equal(free, expected_free)
    n_nodes = 4
    bcs = {1: [False, True, False, False, False, False], 3: [False, False, False, True, False, False]}
    total_dofs = 6 * n_nodes
    expected_fixed = np.array([7, 21], dtype=int)
    all_dofs = np.arange(total_dofs, dtype=int)
    expected_free = np.setdiff1d(all_dofs, expected_fixed)
    (fixed, free) = fcn(boundary_conditions=bcs, n_nodes=n_nodes)
    np.testing.assert_array_equal(fixed, expected_fixed)
    np.testing.assert_array_equal(free, expected_free)

def test_partition_dofs_edge_cases(fcn):
    """Test edge conditions for partition_degrees_of_freedom including empty inputs,
extra or malformed boundary condition entries, and mixed-type flags."""
    n_nodes = 0
    bcs = {}
    expected_fixed = np.array([], dtype=int)
    expected_free = np.array([], dtype=int)
    (fixed, free) = fcn(boundary_conditions=bcs, n_nodes=n_nodes)
    assert fixed.shape == (0,)
    assert free.shape == (0,)
    np.testing.assert_array_equal(fixed, expected_fixed)
    np.testing.assert_array_equal(free, expected_free)
    assert isinstance(fixed, np.ndarray)
    assert isinstance(free, np.ndarray)
    n_nodes = 2
    bcs = {0: [True] * 6, 2: [True] * 6}
    expected_fixed = np.arange(6, dtype=int)
    expected_free = np.arange(6, 12, dtype=int)
    (fixed, free) = fcn(boundary_conditions=bcs, n_nodes=n_nodes)
    np.testing.assert_array_equal(fixed, expected_fixed)
    np.testing.assert_array_equal(free, expected_free)
    n_nodes = 2
    bcs_short = {0: [True, False]}
    with pytest.raises(ValueError):
        fcn(boundary_conditions=bcs_short, n_nodes=n_nodes)
    bcs_long = {0: [True] * 7}
    with pytest.raises(ValueError):
        fcn(boundary_conditions=bcs_long, n_nodes=n_nodes)
    n_nodes = 1
    bcs = {0: [1, 0, 'text', [], (1,), None]}
    expected_fixed = np.array([0, 2, 4], dtype=int)
    expected_free = np.array([1, 3, 5], dtype=int)
    (fixed, free) = fcn(boundary_conditions=bcs, n_nodes=n_nodes)
    np.testing.assert_array_equal(fixed, expected_fixed)
    np.testing.assert_array_equal(free, expected_free)