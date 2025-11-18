def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (coords, connect) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert coords.shape[1] == 2
    assert connect.shape[1] == 8
    expected_elements = 2 * 2
    expected_nodes = (2 * 2 + 1) * (2 * 2 + 1) - 2 * 2
    assert len(connect) == expected_elements
    assert len(coords) == expected_nodes
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    assert np.any(x_coords == 0.0)
    assert np.any(x_coords == 1.0)
    assert np.any(y_coords == 0.0)
    assert np.any(y_coords == 1.0)
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    dx_expected = 0.25
    assert np.allclose(np.diff(unique_x), dx_expected)
    assert np.allclose(np.diff(unique_y), dx_expected)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (coords, connect) = fcn(0.0, 0.0, 2.0, 1.0, 2, 1)
    max_node_id = len(coords) - 1
    assert np.all(connect >= 0)
    assert np.all(connect <= max_node_id)
    for elem in connect:
        assert len(np.unique(elem)) == 8
    for elem in connect:
        corners = elem[:4]
        corner_coords = coords[corners]
        x = corner_coords[:, 0]
        y = corner_coords[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        assert area > 0, 'Corner nodes should form counter-clockwise polygon'
    for elem in connect:
        (n1, n2, n3, n4, n5, n6, n7, n8) = elem
        (n1_coord, n2_coord, n3_coord, n4_coord) = coords[[n1, n2, n3, n4]]
        (n5_coord, n6_coord, n7_coord, n8_coord) = coords[[n5, n6, n7, n8]]
        assert np.allclose(n5_coord, (n1_coord + n2_coord) / 2)
        assert np.allclose(n6_coord, (n2_coord + n3_coord) / 2)
        assert np.allclose(n7_coord, (n3_coord + n4_coord) / 2)
        assert np.allclose(n8_coord, (n4_coord + n1_coord) / 2)
    if len(connect) > 1:
        elem0 = connect[0]
        elem1 = connect[1]
        assert elem0[1] == elem1[0]
        assert elem0[2] == elem1[3]

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
        fcn(2.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)