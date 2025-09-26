def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    import numpy as np
    (coords, connect) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords.shape[0] == 25
    assert connect.shape[0] == 8
    assert coords.shape == (25, 2)
    assert connect.shape == (8, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    assert np.min(x_coords) == 0.0
    assert np.max(x_coords) == 1.0
    assert np.min(y_coords) == 0.0
    assert np.max(y_coords) == 1.0
    dx = 0.5
    dy = 0.5
    expected_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected_y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    for iy in range(5):
        for ix in range(5):
            node_id = iy * 5 + ix
            assert np.allclose(coords[node_id, 0], expected_x[ix])
            assert np.allclose(coords[node_id, 1], expected_y[iy])
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    import numpy as np
    (coords, connect) = fcn(1.0, 2.0, 4.0, 5.0, 3, 2)
    n_nodes = coords.shape[0]
    n_elements = connect.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < n_nodes)
    for elem in connect:
        assert len(np.unique(elem)) == 6
    for elem in connect:
        (n1, n2, n3, n4, n5, n6) = elem
        p1 = coords[n1]
        p2 = coords[n2]
        p3 = coords[n3]
        v1 = p2 - p1
        v2 = p3 - p1
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross > 0, 'Corner nodes not counter-clockwise'
        assert np.allclose(coords[n4], 0.5 * (p1 + p2))
        assert np.allclose(coords[n5], 0.5 * (p2 + p3))
        assert np.allclose(coords[n6], 0.5 * (p3 + p1))
    edges = {}
    for (elem_idx, elem) in enumerate(connect):
        (n1, n2, n3, n4, n5, n6) = elem
        elem_edges = [(min(n1, n2), max(n1, n2), n4), (min(n2, n3), max(n2, n3), n5), (min(n3, n1), max(n3, n1), n6)]
        for edge in elem_edges:
            edge_key = (edge[0], edge[1])
            if edge_key in edges:
                assert edges[edge_key] == edge[2], 'Shared edge has different midside node'
            else:
                edges[edge_key] = edge[2]

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    import pytest
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -1, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, -1)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 2, 2)