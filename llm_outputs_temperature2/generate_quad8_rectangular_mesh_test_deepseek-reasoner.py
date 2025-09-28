def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords1.shape == (21, 2)
    assert connect1.shape == (4, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.any(coords1[:, 0] == 0.0)
    assert np.any(coords1[:, 0] == 1.0)
    assert np.any(coords1[:, 1] == 0.0)
    assert np.any(coords1[:, 1] == 1.0)
    x_coords = np.unique(coords1[:, 0])
    y_coords = np.unique(coords1[:, 1])
    expected_x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    expected_y = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    np.testing.assert_array_almost_equal(x_coords, expected_x)
    np.testing.assert_array_almost_equal(y_coords, expected_y)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    np.testing.assert_array_equal(coords1, coords2)
    np.testing.assert_array_equal(connect1, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (coords, connect) = fcn(0.0, 0.0, 2.0, 1.0, 2, 1)
    n_nodes = coords.shape[0]
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
        np.testing.assert_array_almost_equal(coords[midsides[0]], (coords[corners[0]] + coords[corners[1]]) / 2)
        np.testing.assert_array_almost_equal(coords[midsides[1]], (coords[corners[1]] + coords[corners[2]]) / 2)
        np.testing.assert_array_almost_equal(coords[midsides[2]], (coords[corners[2]] + coords[corners[3]]) / 2)
        np.testing.assert_array_almost_equal(coords[midsides[3]], (coords[corners[3]] + coords[corners[0]]) / 2)
    if connect.shape[0] > 1:
        elem0_nodes = connect[0]
        elem1_nodes = connect[1]
        shared_corners = set(elem0_nodes[[1, 2]]).intersection(elem1_nodes[[0, 3]])
        assert len(shared_corners) == 2
        shared_midside = set(elem0_nodes[[5, 6]]).intersection(elem1_nodes[[4, 7]])
        assert len(shared_midside) == 1

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)