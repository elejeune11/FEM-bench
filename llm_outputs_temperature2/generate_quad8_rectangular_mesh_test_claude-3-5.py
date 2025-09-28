def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements."""
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords1.shape[1] == 2
    assert coords1.dtype == np.float64
    assert connect1.shape == (4, 8)
    assert connect1.dtype == np.int64
    n_nodes = (2 * 2 + 1) * (2 * 2 + 1) - 2 * 2
    assert coords1.shape[0] == n_nodes
    corners = set((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0))
    mesh_corners = set(map(tuple, coords1[connect1[:, :4].flatten()]))
    assert corners.issubset(mesh_corners)
    dx = dy = 0.25
    x_coords = coords1[:, 0]
    y_coords = coords1[:, 1]
    x_diffs = np.diff(np.sort(np.unique(x_coords)))
    y_diffs = np.diff(np.sort(np.unique(y_coords)))
    assert np.allclose(x_diffs, dx)
    assert np.allclose(y_diffs, dy)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements."""
    (coords, connect) = fcn(-1.0, -2.0, 2.0, 1.0, 2, 2)
    assert np.all(connect >= 0)
    assert np.all(connect < len(coords))
    for elem in connect:
        corners = coords[elem[:4]]
        area = 0.5 * np.abs(np.dot(corners[:, 0], np.roll(corners[:, 1], 1)) - np.dot(corners[:, 1], np.roll(corners[:, 0], 1)))
        assert area > 0
    for elem in connect:
        assert np.allclose(coords[elem[4]], (coords[elem[0]] + coords[elem[1]]) / 2)
        assert np.allclose(coords[elem[5]], (coords[elem[1]] + coords[elem[2]]) / 2)
        assert np.allclose(coords[elem[6]], (coords[elem[2]] + coords[elem[3]]) / 2)
        assert np.allclose(coords[elem[7]], (coords[elem[3]] + coords[elem[0]]) / 2)
    edge_nodes = set()
    for elem in connect:
        edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[3]), (elem[3], elem[0])]
        for edge in edges:
            edge = tuple(sorted(edge))
            if edge in edge_nodes:
                edge_nodes.remove(edge)
            else:
                edge_nodes.add(edge)
    assert len(edge_nodes) == 8

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation."""
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