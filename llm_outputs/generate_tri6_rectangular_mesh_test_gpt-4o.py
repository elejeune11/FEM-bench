def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain."""
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords1.shape == ((2 * nx + 1) * (2 * ny + 1), 2)
    assert connect1.shape == (2 * nx * ny, 6)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.allclose(coords1[0], [xl, yl])
    assert np.allclose(coords1[2 * nx], [xh, yl])
    assert np.allclose(coords1[2 * ny * (2 * nx + 1)], [xl, yh])
    assert np.allclose(coords1[-1], [xh, yh])
    (dx, dy) = ((xh - xl) / nx, (yh - yl) / ny)
    for ix in range(2 * nx + 1):
        for iy in range(2 * ny + 1):
            node_id = iy * (2 * nx + 1) + ix
            expected_x = xl + 0.5 * dx * ix
            expected_y = yl + 0.5 * dy * iy
            assert np.allclose(coords1[node_id], [expected_x, expected_y])
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain."""
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (2, 1)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.all(connect >= 0)
    assert np.all(connect < coords.shape[0])
    for elem in connect:
        assert len(set(elem)) == 6
    for elem in connect:
        (N1, N2, N3) = elem[:3]
        (x1, y1) = coords[N1]
        (x2, y2) = coords[N2]
        (x3, y3) = coords[N3]
        assert (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) > 0
    for elem in connect:
        (N1, N2, N3, N4, N5, N6) = elem
        assert np.allclose(coords[N4], (coords[N1] + coords[N2]) / 2)
        assert np.allclose(coords[N5], (coords[N2] + coords[N3]) / 2)
        assert np.allclose(coords[N6], (coords[N3] + coords[N1]) / 2)
    node_pairs = set()
    for elem in connect:
        for i in range(3):
            pair = tuple(sorted((elem[i], elem[(i + 1) % 3])))
            node_pairs.add(pair)
    assert len(node_pairs) == 3 * nx * ny

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs."""
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 1, 1)