def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    xl, yl, xh, yh = (0.0, 0.0, 1.0, 1.0)
    nx, ny = (2, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    hx = 0.5 * dx
    hy = 0.5 * dy
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    Nnodes_expected = npx * npy - nx * ny
    Ne_expected = nx * ny
    assert coords.shape == (Nnodes_expected, 2)
    assert coords.dtype == np.float64
    assert connect.shape == (Ne_expected, 8)
    assert connect.dtype == np.int64
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]], dtype=np.float64)
    for pt in corners:
        assert np.any(np.all(coords == pt, axis=1))
    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    expected_x = xl + hx * np.arange(npx)
    expected_y = yl + hy * np.arange(npy)
    assert np.allclose(unique_x, expected_x)
    assert np.allclose(unique_y, expected_y)
    assert np.allclose(np.diff(unique_x), hx)
    assert np.allclose(np.diff(unique_y), hy)
    coords2, connect2 = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    xl, yl, xh, yh = (-1.0, 0.5, 2.0, 3.0)
    nx, ny = (3, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    Ne = connect.shape[0]
    assert connect.min() >= 0
    assert connect.max() < Nnodes
    for e in range(Ne):
        row = connect[e]
        assert len(np.unique(row)) == 8
    for e in range(Ne):
        N1, N2, N3, N4 = (connect[e, 0], connect[e, 1], connect[e, 2], connect[e, 3])
        poly = np.vstack([coords[N1], coords[N2], coords[N3], coords[N4]])
        x = poly[:, 0]
        y = poly[:, 1]
        area2 = np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))
        assert area2 > 0.0
    for e in range(Ne):
        N1, N2, N3, N4, N5, N6, N7, N8 = connect[e]
        c1, c2, c3, c4 = (coords[N1], coords[N2], coords[N3], coords[N4])
        m5, m6, m7, m8 = (coords[N5], coords[N6], coords[N7], coords[N8])
        assert np.allclose(m5, 0.5 * (c1 + c2))
        assert np.allclose(m6, 0.5 * (c2 + c3))
        assert np.allclose(m7, 0.5 * (c3 + c4))
        assert np.allclose(m8, 0.5 * (c4 + c1))
    for cy in range(ny):
        for cx in range(nx - 1):
            e = cy * nx + cx
            eR = e + 1
            assert connect[e, 1] == connect[eR, 0]
            assert connect[e, 2] == connect[eR, 3]
            assert connect[e, 5] == connect[eR, 7]
    for cy in range(ny - 1):
        for cx in range(nx):
            e = cy * nx + cx
            eU = e + nx
            assert connect[e, 3] == connect[eU, 0]
            assert connect[e, 2] == connect[eU, 1]
            assert connect[e, 6] == connect[eU, 4]

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
        fcn(2.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)