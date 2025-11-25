def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (coords1, connect1) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    (coords2, connect2) = fcn(0.0, 0.0, 1.0, 1.0, 2, 2)
    np.testing.assert_array_equal(coords1, coords2)
    np.testing.assert_array_equal(connect1, connect2)
    assert coords1.shape == (21, 2)
    assert connect1.shape == (4, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.min(coords1[:, 0]) == 0.0
    assert np.max(coords1[:, 0]) == 1.0
    assert np.min(coords1[:, 1]) == 0.0
    assert np.max(coords1[:, 1]) == 1.0
    dx = 0.5 / 2
    dy = 0.5 / 2
    x_coords = coords1[:, 0]
    y_coords = coords1[:, 1]
    assert np.allclose(x_coords % dx, 0, atol=1e-10) or np.allclose(x_coords % dx, dx, atol=1e-10)
    assert np.allclose(y_coords % dy, 0, atol=1e-10) or np.allclose(y_coords % dy, dy, atol=1e-10)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (coords, connect) = fcn(0.0, 0.0, 2.0, 1.0, 2, 1)
    n_nodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < n_nodes)
    for i in range(connect.shape[0]):
        assert len(np.unique(connect[i])) == 8
    for elem in connect:
        corners = elem[:4]
        corner_coords = coords[corners]
        x = corner_coords[:, 0]
        y = corner_coords[:, 1]
        area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
        assert area > 0, 'Corner nodes should form counter-clockwise polygon'
    for elem in connect:
        corners = elem[:4]
        midsides = elem[4:]
        corner_coords = coords[corners]
        midside_coords = coords[midsides]
        expected_N5 = (corner_coords[0] + corner_coords[1]) / 2
        np.testing.assert_allclose(midside_coords[0], expected_N5, atol=1e-10)
        expected_N6 = (corner_coords[1] + corner_coords[2]) / 2
        np.testing.assert_allclose(midside_coords[1], expected_N6, atol=1e-10)
        expected_N7 = (corner_coords[2] + corner_coords[3]) / 2
        np.testing.assert_allclose(midside_coords[2], expected_N7, atol=1e-10)
        expected_N8 = (corner_coords[3] + corner_coords[0]) / 2
        np.testing.assert_allclose(midside_coords[3], expected_N8, atol=1e-10)
    if connect.shape[0] > 1:
        all_nodes = connect.flatten()
        unique_nodes = np.unique(all_nodes)
        assert len(unique_nodes) < len(all_nodes), 'Elements should share nodes'

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