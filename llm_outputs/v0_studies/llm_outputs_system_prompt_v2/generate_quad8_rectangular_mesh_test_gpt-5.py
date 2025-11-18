def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_elements = nx * ny
    assert isinstance(coords, np.ndarray)
    assert isinstance(connect, np.ndarray)
    assert coords.shape == (expected_nodes, 2)
    assert coords.dtype == np.float64
    assert connect.shape == (expected_elements, 8)
    assert connect.dtype == np.int64
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]], dtype=np.float64)
    for c in corners:
        assert np.any(np.all(np.isclose(coords, c, atol=1e-12), axis=1))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    expected_xs = np.linspace(xl, xh, 2 * nx + 1, dtype=np.float64)
    expected_ys = np.linspace(yl, yh, 2 * ny + 1, dtype=np.float64)
    ux = np.unique(coords[:, 0])
    uy = np.unique(coords[:, 1])
    assert np.allclose(ux, expected_xs, atol=0, rtol=0)
    assert np.allclose(uy, expected_ys, atol=0, rtol=0)
    centers = []
    for ix in range(1, 2 * nx, 2):
        x = xl + 0.5 * dx * ix
        for iy in range(1, 2 * ny, 2):
            y = yl + 0.5 * dy * iy
            centers.append([x, y])
    centers = np.array(centers, dtype=np.float64)
    for c in centers:
        assert not np.any(np.all(np.isclose(coords, c, atol=1e-12), axis=1))
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 2.0, 5.0, 7.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    Ne = nx * ny
    assert connect.shape == (Ne, 8)
    assert connect.dtype == np.int64
    assert coords.shape[0] == Nnodes
    assert connect.min() >= 0
    assert connect.max() < Nnodes
    for e in range(Ne):
        row = connect[e]
        assert len(np.unique(row)) == 8
    c4 = coords[connect[:, 0:4]]
    x = c4[:, :, 0]
    y = c4[:, :, 1]
    area2 = x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0] + (x[:, 1] * y[:, 2] - x[:, 2] * y[:, 1]) + (x[:, 2] * y[:, 3] - x[:, 3] * y[:, 2]) + (x[:, 3] * y[:, 0] - x[:, 0] * y[:, 3])
    assert np.all(area2 > 0.0)
    mids = coords[connect[:, 4:8]]
    expected_mids = np.empty_like(mids)
    expected_mids[:, 0, :] = 0.5 * (c4[:, 0, :] + c4[:, 1, :])
    expected_mids[:, 1, :] = 0.5 * (c4[:, 1, :] + c4[:, 2, :])
    expected_mids[:, 2, :] = 0.5 * (c4[:, 2, :] + c4[:, 3, :])
    expected_mids[:, 3, :] = 0.5 * (c4[:, 3, :] + c4[:, 0, :])
    assert np.allclose(mids, expected_mids, atol=0, rtol=0)
    for cy in range(ny):
        for cx in range(nx):
            e = cy * nx + cx
            row = connect[e]
            if cx < nx - 1:
                er = cy * nx + (cx + 1)
                row_r = connect[er]
                assert row[1] == row_r[0]
                assert row[2] == row_r[3]
            if cy < ny - 1:
                et = (cy + 1) * nx + cx
                row_t = connect[et]
                assert row[3] == row_t[0]
                assert row[2] == row_t[1]

def test_quad8_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs in Quad8 mesh generation.
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