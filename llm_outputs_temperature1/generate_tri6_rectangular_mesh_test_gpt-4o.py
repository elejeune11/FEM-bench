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
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    expected_coords = np.array([[xl + 0.5 * dx * ix, yl + 0.5 * dy * iy] for iy in range(2 * ny + 1) for ix in range(2 * nx + 1)])
    assert np.allclose(coords1, expected_coords)
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain."""
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (2, 1)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.all((connect >= 0) & (connect < coords.shape[0]))
    for elem in connect:
        (x1, y1) = coords[elem[0]]
        (x2, y2) = coords[elem[1]]
        (x3, y3) = coords[elem[2]]
        assert (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) > 0
    for elem in connect:
        (x1, y1) = coords[elem[0]]
        (x2, y2) = coords[elem[1]]
        (x3, y3) = coords[elem[2]]
        (x4, y4) = coords[elem[3]]
        (x5, y5) = coords[elem[4]]
        (x6, y6) = coords[elem[5]]
        assert np.allclose([x4, y4], [(x1 + x2) / 2, (y1 + y2) / 2])
        assert np.allclose([x5, y5], [(x2 + x3) / 2, (y2 + y3) / 2])
        assert np.allclose([x6, y6], [(x3 + x1) / 2, (y3 + y1) / 2])
    node_pairs = set()
    for elem in connect:
        node_pairs.update([(elem[i], elem[j]) for (i, j) in [(0, 1), (1, 2), (2, 0)]])
    assert len(node_pairs) == len(connect) * 3 // 2

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