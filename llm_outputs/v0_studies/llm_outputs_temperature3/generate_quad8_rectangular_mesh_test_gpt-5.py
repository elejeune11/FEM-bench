def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    expected_nodes = npx * npy - nx * ny
    expected_elems = nx * ny
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elems, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    corners = {(xl, yl), (xh, yl), (xh, yh), (xl, yh)}
    coords_set = set(map(tuple, coords))
    assert corners.issubset(coords_set)
    xs = coords[:, 0]
    ys = coords[:, 1]
    assert np.isclose(xs.min(), xl) and np.isclose(xs.max(), xh)
    assert np.isclose(ys.min(), yl) and np.isclose(ys.max(), yh)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    hx = 0.5 * dx
    hy = 0.5 * dy
    unique_x = np.unique(xs)
    unique_y = np.unique(ys)
    assert len(unique_x) == npx
    assert len(unique_y) == npy
    if len(unique_x) > 1:
        assert np.allclose(np.diff(unique_x), hx)
    if len(unique_y) > 1:
        assert np.allclose(np.diff(unique_y), hy)
    ix = np.arange(npx)
    iy = np.arange(npy)
    (IX, IY) = np.meshgrid(ix, iy, indexing='xy')
    X = xl + hx * IX
    Y = yl + hy * IY
    mask = ~((IX % 2 == 1) & (IY % 2 == 1))
    expected_coords = np.column_stack([X.ravel(order='C')[mask.ravel(order='C')], Y.ravel(order='C')[mask.ravel(order='C')]])
    assert np.array_equal(coords, expected_coords)
    ix_from_coords = np.rint((coords[:, 0] - xl) / hx).astype(int)
    iy_from_coords = np.rint((coords[:, 1] - yl) / hy).astype(int)
    assert not np.any((ix_from_coords % 2 == 1) & (iy_from_coords % 2 == 1))
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.2, 0.3, 2.4, 3.1)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    Ne = nx * ny
    assert connect.shape == (Ne, 8)
    assert connect.min() >= 0 and connect.max() < Nnodes
    for e in range(Ne):
        row = connect[e]
        assert len(set(row.tolist())) == 8

    def polygon_area_xy(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
    for e in range(Ne):
        n = connect[e]
        corners = coords[n[[0, 1, 2, 3]], :]
        area = polygon_area_xy(corners)
        assert area > 0.0
    for e in range(Ne):
        n = connect[e]
        (c1, c2, c3, c4) = (coords[n[0]], coords[n[1]], coords[n[2]], coords[n[3]])
        (m5, m6, m7, m8) = (coords[n[4]], coords[n[5]], coords[n[6]], coords[n[7]])
        assert np.allclose(m5, 0.5 * (c1 + c2), rtol=0.0, atol=1e-15)
        assert np.allclose(m6, 0.5 * (c2 + c3), rtol=0.0, atol=1e-15)
        assert np.allclose(m7, 0.5 * (c3 + c4), rtol=0.0, atol=1e-15)
        assert np.allclose(m8, 0.5 * (c4 + c1), rtol=0.0, atol=1e-15)
    for cy in range(ny):
        for cx in range(nx - 1):
            eL = cy * nx + cx
            eR = cy * nx + (cx + 1)
            left_edge_right = set(connect[eL, [1, 5, 2]].tolist())
            right_edge_left = set(connect[eR, [0, 7, 3]].tolist())
            assert left_edge_right == right_edge_left
    for cy in range(ny - 1):
        for cx in range(nx):
            eB = cy * nx + cx
            eT = (cy + 1) * nx + cx
            bottom_top_edge = set(connect[eB, [2, 6, 3]].tolist())
            top_bottom_edge = set(connect[eT, [1, 4, 0]].tolist())
            assert bottom_top_edge == top_bottom_edge

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
        fcn(2.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)