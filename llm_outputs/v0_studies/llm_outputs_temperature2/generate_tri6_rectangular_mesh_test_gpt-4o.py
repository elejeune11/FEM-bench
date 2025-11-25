def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh, nx, ny) = (0.0, 0.0, 1.0, 1.0, 2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords.shape == ((2 * nx + 1) * (2 * ny + 1), 2)
    assert connect.shape == (2 * nx * ny, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.allclose(coords[0], [xl, yl])
    assert np.allclose(coords[2 * nx], [xh, yl])
    assert np.allclose(coords[-1], [xh, yh])
    assert np.allclose(coords[-1 - 2 * nx], [xl, yh])
    (dx, dy) = ((xh - xl) / nx, (yh - yl) / ny)
    expected_coords = np.array([[xl + 0.5 * dx * ix, yl + 0.5 * dy * iy] for iy in range(2 * ny + 1) for ix in range(2 * nx + 1)])
    assert np.allclose(coords, expected_coords)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh, nx, ny) = (0.0, 0.0, 2.0, 1.0, 2, 1)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.all(connect >= 0)
    assert np.all(connect < coords.shape[0])
    assert all((len(set(element)) == 6 for element in connect))
    for element in connect:
        (N1, N2, N3) = element[:3]
        (x1, y1) = coords[N1]
        (x2, y2) = coords[N2]
        (x3, y3) = coords[N3]
        assert (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) > 0
    for element in connect:
        (N1, N2, N3, N4, N5, N6) = element
        assert np.allclose(coords[N4], (coords[N1] + coords[N2]) / 2)
        assert np.allclose(coords[N5], (coords[N2] + coords[N3]) / 2)
        assert np.allclose(coords[N6], (coords[N3] + coords[N1]) / 2)
    node_pairs = [(0, 1), (1, 2), (2, 0)]
    for (element1, element2) in zip(connect[::2], connect[1::2]):
        for (i, j) in node_pairs:
            assert element1[i] in element2 or element1[j] in element2

def test_tri6_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 1, 1)