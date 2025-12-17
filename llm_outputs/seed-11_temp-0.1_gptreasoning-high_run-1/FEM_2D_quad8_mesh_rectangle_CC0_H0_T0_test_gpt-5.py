def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    xl, yl, xh, yh = (0.0, 0.0, 1.0, 1.0)
    nx, ny = (2, 2)
    coords1, connect1 = fcn(xl, yl, xh, yh, nx, ny)
    expected_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_elems = nx * ny
    assert coords1.shape == (expected_nodes, 2)
    assert connect1.shape == (expected_elems, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.isclose(coords1[:, 0].min(), xl)
    assert np.isclose(coords1[:, 0].max(), xh)
    assert np.isclose(coords1[:, 1].min(), yl)
    assert np.isclose(coords1[:, 1].max(), yh)
    for cx, cy in [(xl, yl), (xh, yl), (xh, yh), (xl, yh)]:
        mask = np.isclose(coords1[:, 0], cx) & np.isclose(coords1[:, 1], cy)
        assert mask.sum() == 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    hx = dx / 2
    hy = dy / 2
    ux = np.unique(coords1[:, 0])
    uy = np.unique(coords1[:, 1])
    assert ux.size == 2 * nx + 1
    assert uy.size == 2 * ny + 1
    assert np.allclose(np.diff(ux), hx)
    assert np.allclose(np.diff(uy), hy)
    coords2, connect2 = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    xl, yl, xh, yh = (-1.0, -2.0, 4.0, 1.0)
    nx, ny = (3, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    Ne = nx * ny
    assert connect.shape == (Ne, 8)
    assert connect.dtype == np.int64
    assert connect.min() >= 0
    assert connect.max() < Nnodes
    for e in range(Ne):
        assert np.unique(connect[e]).size == 8
    corners = coords[connect[:, 0:4]]
    x = corners[:, :, 0]
    y = corners[:, :, 1]
    area2 = np.sum(x * np.roll(y, -1, axis=1) - y * np.roll(x, -1, axis=1), axis=1)
    assert np.all(area2 > 0)
    c = connect
    cb = coords
    assert np.allclose(cb[c[:, 4]], 0.5 * (cb[c[:, 0]] + cb[c[:, 1]]))
    assert np.allclose(cb[c[:, 5]], 0.5 * (cb[c[:, 1]] + cb[c[:, 2]]))
    assert np.allclose(cb[c[:, 6]], 0.5 * (cb[c[:, 2]] + cb[c[:, 3]]))
    assert np.allclose(cb[c[:, 7]], 0.5 * (cb[c[:, 3]] + cb[c[:, 0]]))
    for cy in range(ny):
        for cx in range(nx - 1):
            e = cy * nx + cx
            er = cy * nx + (cx + 1)
            assert c[e, 1] == c[er, 0]
            assert c[e, 2] == c[er, 3]
    for cy in range(ny - 1):
        for cx in range(nx):
            e = cy * nx + cx
            et = (cy + 1) * nx + cx
            assert c[e, 3] == c[et, 0]
            assert c[e, 2] == c[et, 1]

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