def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements."""
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_elements = nx * ny
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elements, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert any(np.all(coords == [xl, yl], axis=1))
    assert any(np.all(coords == [xh, yl], axis=1))
    assert any(np.all(coords == [xh, yh], axis=1))
    assert any(np.all(coords == [xl, yh], axis=1))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    assert np.allclose(np.diff(unique_x), 0.5 * dx)
    assert np.allclose(np.diff(unique_y), 0.5 * dy)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements."""
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (2, 1)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.all(connect >= 0)
    assert np.all(connect < coords.shape[0])
    for elem in connect:
        x_coords = coords[elem[:4], 0]
        y_coords = coords[elem[:4], 1]
        area = 0.5 * np.sum(x_coords[:-1] * y_coords[1:] - x_coords[1:] * y_coords[:-1])
        assert area > 0
    for elem in connect:
        (N1, N2, N3, N4, N5, N6, N7, N8) = elem
        assert np.allclose(coords[N5], (coords[N1] + coords[N2]) / 2)
        assert np.allclose(coords[N6], (coords[N2] + coords[N3]) / 2)
        assert np.allclose(coords[N7], (coords[N3] + coords[N4]) / 2)
        assert np.allclose(coords[N8], (coords[N4] + coords[N1]) / 2)

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation."""
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 1, 1)