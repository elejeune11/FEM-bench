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
    assert connect.shape == (Ne_expected, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    xs = [xl + i * hx for i in range(npx)]
    ys = [yl + j * hy for j in range(npy)]
    uniq_x = np.unique(coords[:, 0])
    uniq_y = np.unique(coords[:, 1])
    assert np.allclose(uniq_x, xs)
    assert np.allclose(uniq_y, ys)
    assert np.isclose(coords[:, 0].min(), xl)
    assert np.isclose(coords[:, 0].max(), xh)
    assert np.isclose(coords[:, 1].min(), yl)
    assert np.isclose(coords[:, 1].max(), yh)
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]])
    for c in corners:
        assert np.any(np.all(np.isclose(coords, c), axis=1))
    for x, y in coords:
        kx = int(round((x - xl) / hx))
        ky = int(round((y - yl) / hy))
        assert np.isclose(x, xl + kx * hx)
        assert np.isclose(y, yl + ky * hy)
        assert not (kx % 2 == 1 and ky % 2 == 1)
    id_map = -np.ones((npy, npx), dtype=np.int64)
    coords_expected = []
    nid = 0
    for iy in range(npy):
        for ix in range(npx):
            if ix % 2 == 1 and iy % 2 == 1:
                continue
            coords_expected.append([xl + hx * ix, yl + hy * iy])
            id_map[iy, ix] = nid
            nid += 1
    coords_expected = np.asarray(coords_expected, dtype=np.float64)
    assert np.array_equal(coords, coords_expected)
    connect_expected = []
    for cy in range(ny):
        for cx in range(nx):
            ix0, iy0 = (2 * cx, 2 * cy)
            N1 = id_map[iy0, ix0]
            N2 = id_map[iy0, ix0 + 2]
            N3 = id_map[iy0 + 2, ix0 + 2]
            N4 = id_map[iy0 + 2, ix0]
            N5 = id_map[iy0, ix0 + 1]
            N6 = id_map[iy0 + 1, ix0 + 2]
            N7 = id_map[iy0 + 2, ix0 + 1]
            N8 = id_map[iy0 + 1, ix0]
            connect_expected.append([N1, N2, N3, N4, N5, N6, N7, N8])
    connect_expected = np.asarray(connect_expected, dtype=np.int64)
    assert np.array_equal(connect, connect_expected)
    coords2, connect2 = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    xl, yl, xh, yh = (-2.0, 1.0, 3.0, 5.0)
    nx, ny = (3, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    Ne = connect.shape[0]
    assert Ne == nx * ny
    assert connect.min() >= 0
    assert connect.max() < Nnodes
    for e in range(Ne):
        row = connect[e]
        assert len(np.unique(row)) == 8

    def poly_area(p):
        return 0.5 * np.sum(p[:, 0] * np.roll(p[:, 1], -1) - p[:, 1] * np.roll(p[:, 0], -1))
    for e in range(Ne):
        corners = coords[connect[e, 0:4]]
        area = poly_area(corners)
        assert area > 0
    for e in range(Ne):
        p1 = coords[connect[e, 0]]
        p2 = coords[connect[e, 1]]
        p3 = coords[connect[e, 2]]
        p4 = coords[connect[e, 3]]
        p5 = coords[connect[e, 4]]
        p6 = coords[connect[e, 5]]
        p7 = coords[connect[e, 6]]
        p8 = coords[connect[e, 7]]
        assert np.allclose(p5, 0.5 * (p1 + p2), rtol=1e-13, atol=1e-13)
        assert np.allclose(p6, 0.5 * (p2 + p3), rtol=1e-13, atol=1e-13)
        assert np.allclose(p7, 0.5 * (p3 + p4), rtol=1e-13, atol=1e-13)
        assert np.allclose(p8, 0.5 * (p4 + p1), rtol=1e-13, atol=1e-13)
    for cy in range(ny):
        for cx in range(nx - 1):
            idx_left = cy * nx + cx
            idx_right = cy * nx + (cx + 1)
            left_edge = {connect[idx_left, 1], connect[idx_left, 2]}
            right_edge = {connect[idx_right, 0], connect[idx_right, 3]}
            assert left_edge == right_edge
    for cy in range(ny - 1):
        for cx in range(nx):
            idx_below = cy * nx + cx
            idx_above = (cy + 1) * nx + cx
            below_top = {connect[idx_below, 2], connect[idx_below, 3]}
            above_bottom = {connect[idx_above, 0], connect[idx_above, 1]}
            assert below_top == above_bottom

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -3, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, -1)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, -1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 5.0, 1.0, -3.0, 1, 1)