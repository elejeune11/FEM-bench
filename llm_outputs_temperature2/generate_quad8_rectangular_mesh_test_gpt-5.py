def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    expected_nodes = npx * npy - nx * ny
    expected_elements = nx * ny
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elements, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    (xs, ys) = (coords[:, 0], coords[:, 1])
    assert np.isclose(xs.min(), xl)
    assert np.isclose(xs.max(), xh)
    assert np.isclose(ys.min(), yl)
    assert np.isclose(ys.max(), yh)
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]], dtype=np.float64)
    for c in corners:
        assert np.any(np.all(np.isclose(coords, c), axis=1))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    hx = 0.5 * dx
    hy = 0.5 * dy
    rx = (xs - xl) / hx
    ry = (ys - yl) / hy
    assert np.allclose(rx, np.round(rx), atol=1e-12)
    assert np.allclose(ry, np.round(ry), atol=1e-12)
    centers = np.array([[xl + dx * 0.5, yl + dy * 0.5], [xl + dx * 1.5, yl + dy * 0.5], [xl + dx * 0.5, yl + dy * 1.5], [xl + dx * 1.5, yl + dy * 1.5]], dtype=np.float64)
    for c in centers:
        assert not np.any(np.all(np.isclose(coords, c), axis=1))
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (2.0, -1.0, 6.0, 3.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    for elem in connect:
        assert elem.min() >= 0 and elem.max() < Nnodes
        assert len(np.unique(elem)) == 8

    def polygon_area_xy(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - y * np.roll(x, -1))
    for elem in connect:
        (n1, n2, n3, n4, n5, n6, n7, n8) = elem
        corners = coords[[n1, n2, n3, n4], :]
        area = polygon_area_xy(corners)
        assert area > 0.0
        mid_targets = np.array([0.5 * (coords[n1] + coords[n2]), 0.5 * (coords[n2] + coords[n3]), 0.5 * (coords[n3] + coords[n4]), 0.5 * (coords[n4] + coords[n1])])
        mids = coords[[n5, n6, n7, n8], :]
        assert np.allclose(mids, mid_targets, rtol=1e-12, atol=1e-12)

    def elem_index(cx, cy):
        return cy * nx + cx
    for cy in range(ny):
        for cx in range(nx - 1):
            e_left = connect[elem_index(cx, cy)]
            e_right = connect[elem_index(cx + 1, cy)]
            assert e_left[1] == e_right[0]
            assert e_left[2] == e_right[3]
    for cy in range(ny - 1):
        for cx in range(nx):
            e_bottom = connect[elem_index(cx, cy)]
            e_top = connect[elem_index(cx, cy + 1)]
            assert e_bottom[2] == e_top[1]
            assert e_bottom[3] == e_top[0]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -2, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, -3)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, -1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 5.0, 1.0, -2.0, 1, 1)