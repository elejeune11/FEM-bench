def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (coords, connect) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords.shape == (21, 2)
    assert connect.shape == (4, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.any(coords[:, 0] == 0.0)
    assert np.any(coords[:, 0] == 1.0)
    assert np.any(coords[:, 1] == 0.0)
    assert np.any(coords[:, 1] == 1.0)
    x_coords = np.unique(coords[:, 0])
    y_coords = np.unique(coords[:, 1])
    expected_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected_y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    assert np.allclose(x_coords, expected_x)
    assert np.allclose(y_coords, expected_y)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (coords, connect) = fcn(0.0, 0.0, 2.0, 1.0, 2, 1)
    n_nodes = len(coords)
    assert np.all(connect >= 0)
    assert np.all(connect < n_nodes)
    for elem_nodes in connect:
        assert len(np.unique(elem_nodes)) == 8
    for elem_nodes in connect:
        corners = coords[elem_nodes[:4]]
        (x, y) = (corners[:, 0], corners[:, 1])
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        assert area > 0
    for elem_nodes in connect:
        corners = elem_nodes[:4]
        midsides = elem_nodes[4:]
        assert np.allclose(coords[midsides[0]], (coords[corners[0]] + coords[corners[1]]) / 2)
        assert np.allclose(coords[midsides[1]], (coords[corners[1]] + coords[corners[2]]) / 2)
        assert np.allclose(coords[midsides[2]], (coords[corners[2]] + coords[corners[3]]) / 2)
        assert np.allclose(coords[midsides[3]], (coords[corners[3]] + coords[corners[0]]) / 2)
    elem0_nodes = connect[0]
    elem1_nodes = connect[1]
    assert elem0_nodes[1] == elem1_nodes[3]
    assert elem0_nodes[2] == elem1_nodes[0]
    assert elem0_nodes[5] == elem1_nodes[7]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, -1)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)