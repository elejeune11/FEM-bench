def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements."""
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)
    assert coords1.shape == ((2 * nx + 1) * (2 * ny + 1) - nx * ny, 2)
    assert connect1.shape == (nx * ny, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.any(np.all(coords1 == [xl, yl], axis=1))
    assert np.any(np.all(coords1 == [xh, yl], axis=1))
    assert np.any(np.all(coords1 == [xh, yh], axis=1))
    assert np.any(np.all(coords1 == [xl, yh], axis=1))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    unique_x = np.unique(coords1[:, 0])
    unique_y = np.unique(coords1[:, 1])
    assert np.allclose(np.diff(unique_x), dx / 2)
    assert np.allclose(np.diff(unique_y), dy / 2)

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
    for i in range(connect.shape[0] - 1):
        for j in range(i + 1, connect.shape[0]):
            shared_nodes = set(connect[i]) & set(connect[j])
            assert len(shared_nodes) in {0, 2, 4}

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