def test_partition_dofs_correctness(fcn):
    boundary_conditions = {}
    n_nodes = 2
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 0
    assert len(free) == 12
    assert set(free) == set(range(12))
    boundary_conditions = {0: [True] * 6, 1: [True] * 6}
    n_nodes = 2
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 12
    assert len(free) == 0
    assert set(fixed) == set(range(12))
    boundary_conditions = {0: [True, False, True, False, True, False]}
    n_nodes = 2
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 3
    assert len(free) == 9
    assert set(fixed) == {0, 2, 4}
    assert set(free) == set(range(12)) - {0, 2, 4}
    boundary_conditions = {1: [True, True, False, False, False, False], 3: [False, False, True, True, False, False]}
    n_nodes = 4
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 4
    assert len(free) == 20
    assert set(fixed) == {6, 7, 20, 21}
    assert set(free) == set(range(24)) - {6, 7, 20, 21}
    boundary_conditions = {0: [True, False, False, True, False, False], 2: [False, True, False, False, True, False]}
    n_nodes = 3
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 4
    assert len(free) == 14
    assert set(fixed) == {0, 3, 12, 14}
    assert set(free) == set(range(18)) - {0, 3, 12, 14}
    boundary_conditions = {0: [False, False, False, False, False, True]}
    n_nodes = 1
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 1
    assert len(free) == 5
    assert fixed[0] == 5
    assert set(free) == {0, 1, 2, 3, 4}
    boundary_conditions = {}
    n_nodes = 0
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 0
    assert len(free) == 0
    boundary_conditions = {i: [True, False, True, False, True, False] for i in range(3)}
    n_nodes = 3
    (fixed, free) = fcn(boundary_conditions, n_nodes)
    assert len(fixed) == 9
    assert len(free) == 9
    assert set(fixed) == {0, 2, 4, 6, 8, 10, 12, 14, 16}
    assert set(free) == {1, 3, 5, 7, 9, 11, 13, 15, 17}