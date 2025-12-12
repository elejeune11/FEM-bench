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
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    hx = 0.5 * dx
    hy = 0.5 * dy
    xs = xl + hx * np.arange(npx, dtype=np.float64)
    ys = yl + hy * np.arange(npy, dtype=np.float64)
    (IX, IY) = np.meshgrid(np.arange(npx), np.arange(npy), indexing='xy')
    (X, Y) = np.meshgrid(xs, ys, indexing='xy')
    mask = ~((IX % 2 == 1) & (IY % 2 == 1))
    expected_coords = np.column_stack((X.ravel(order='C')[mask.ravel(order='C')], Y.ravel(order='C')[mask.ravel(order='C')]))
    assert np.array_equal(coords, expected_coords)
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]], dtype=np.float64)
    for corner in corners:
        assert np.any(np.all(coords == corner, axis=1))
    ux = np.unique(coords[:, 0])
    uy = np.unique(coords[:, 1])
    if ux.size > 1:
        np.testing.assert_allclose(np.diff(ux), hx, rtol=0, atol=1e-15)
    if uy.size > 1:
        np.testing.assert_allclose(np.diff(uy), hy, rtol=0, atol=1e-15)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (-3.0, 1.0, 4.0, 5.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    for e in range(connect.shape[0]):
        elem = connect[e]
        assert elem.min() >= 0 and elem.max() < Nnodes
        assert np.unique(elem).size == 8
    for e in range(connect.shape[0]):
        (n1, n2, n3, n4) = (connect[e, 0], connect[e, 1], connect[e, 2], connect[e, 3])
        xy = coords[[n1, n2, n3, n4]]
        (x, y) = (xy[:, 0], xy[:, 1])
        area2 = np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        assert area2 > 0.0
    for e in range(connect.shape[0]):
        (N1, N2, N3, N4, N5, N6, N7, N8) = connect[e]
        np.testing.assert_allclose(coords[N5], 0.5 * (coords[N1] + coords[N2]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(coords[N6], 0.5 * (coords[N2] + coords[N3]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(coords[N7], 0.5 * (coords[N3] + coords[N4]), rtol=0, atol=1e-12)
        np.testing.assert_allclose(coords[N8], 0.5 * (coords[N4] + coords[N1]), rtol=0, atol=1e-12)
    for cy in range(ny):
        for cx in range(nx - 1):
            eL = cy * nx + cx
            eR = cy * nx + (cx + 1)
            edge_right_left = connect[eL, [1, 5, 2]]
            edge_left_right = connect[eR, [0, 7, 3]]
            assert np.array_equal(np.sort(edge_right_left), np.sort(edge_left_right))
    for cy in range(ny - 1):
        for cx in range(nx):
            eB = cy * nx + cx
            eT = (cy + 1) * nx + cx
            edge_top_bottom = connect[eB, [2, 6, 3]]
            edge_bottom_top = connect[eT, [0, 4, 1]]
            assert np.array_equal(np.sort(edge_top_bottom), np.sort(edge_bottom_top))

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 0, 1)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, -2, 1)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 1, 0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 1, -3)
    with pytest.raises(ValueError):
        fcn(1.0, yl, 1.0, yh, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, yl, 1.0, yh, 1, 1)
    with pytest.raises(ValueError):
        fcn(xl, 1.0, xh, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(xl, 2.0, xh, 1.0, 1, 1)