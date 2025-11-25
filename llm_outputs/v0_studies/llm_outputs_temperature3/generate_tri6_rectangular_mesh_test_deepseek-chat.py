def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (coords, connect) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords.shape == (25, 2)
    assert connect.shape == (8, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.min(coords[:, 0]) == 0.0
    assert np.max(coords[:, 0]) == 1.0
    assert np.min(coords[:, 1]) == 0.0
    assert np.max(coords[:, 1]) == 1.0
    x_coords = np.unique(coords[:, 0])
    y_coords = np.unique(coords[:, 1])
    dx_expected = 0.25
    dy_expected = 0.25
    assert np.allclose(np.diff(x_coords), dx_expected)
    assert np.allclose(np.diff(y_coords), dy_expected)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (coords, connect) = fcn(0.0, 0.0, 2.0, 3.0, 2, 3)
    n_nodes = coords.shape[0]
    assert np.min(connect) >= 0
    assert np.max(connect) < n_nodes
    for elem in connect:
        assert len(np.unique(elem)) == 6
    for elem in connect:
        (n1, n2, n3) = (coords[elem[0]], coords[elem[1]], coords[elem[2]])
        cross = (n2[0] - n1[0]) * (n3[1] - n1[1]) - (n2[1] - n1[1]) * (n3[0] - n1[0])
        assert cross > 0
    for elem in connect:
        (n1, n2, n3, n4, n5, n6) = elem
        (c1, c2, c3) = (coords[n1], coords[n2], coords[n3])
        (m4, m5, m6) = (coords[n4], coords[n5], coords[n6])
        assert np.allclose(m4, (c1 + c2) / 2)
        assert np.allclose(m5, (c2 + c3) / 2)
        assert np.allclose(m6, (c3 + c1) / 2)
    edge_dict = {}
    for (i, elem) in enumerate(connect):
        edges = [tuple(sorted([elem[0], elem[1], elem[3]])), tuple(sorted([elem[1], elem[2], elem[4]])), tuple(sorted([elem[2], elem[0], elem[5]]))]
        for edge in edges:
            if edge in edge_dict:
                assert edge_dict[edge] == i, f'Edge {edge} shared between elements {edge_dict[edge]} and {i}'
            else:
                edge_dict[edge] = i

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
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