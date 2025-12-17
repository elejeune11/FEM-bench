def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    xl, yl, xh, yh = (0.0, 0.0, 1.0, 1.0)
    nx, ny = (2, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    npx, npy = (2 * nx + 1, 2 * ny + 1)
    Nnodes = npx * npy
    Ne = 2 * nx * ny
    assert coords.shape == (Nnodes, 2)
    assert coords.dtype == np.float64
    assert connect.shape == (Ne, 6)
    assert connect.dtype == np.int64
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    hx = 0.5 * dx
    hy = 0.5 * dy
    ix = np.arange(npx)
    iy = np.arange(npy)
    X, Y = np.meshgrid(ix, iy, indexing='xy')
    X = xl + hx * X.astype(np.float64)
    Y = yl + hy * Y.astype(np.float64)
    expected_coords = np.column_stack([X.ravel(order='C'), Y.ravel(order='C')])
    assert np.allclose(coords, expected_coords)
    assert np.isclose(coords[:, 0].min(), xl)
    assert np.isclose(coords[:, 0].max(), xh)
    assert np.isclose(coords[:, 1].min(), yl)
    assert np.isclose(coords[:, 1].max(), yh)
    for corner in [(xl, yl), (xh, yl), (xl, yh), (xh, yh)]:
        matches = np.where(np.isclose(coords[:, 0], corner[0]) & np.isclose(coords[:, 1], corner[1]))[0]
        assert matches.size == 1
    coords2, connect2 = fcn(xl, yl, xh, yh, nx, ny)
    assert coords2.shape == coords.shape and connect2.shape == connect.shape
    assert coords2.dtype == coords.dtype and connect2.dtype == connect.dtype
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    xl, yl, xh, yh = (-1.5, 2.0, 2.5, 5.0)
    nx, ny = (3, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < Nnodes
    for elem in connect:
        assert len(np.unique(elem)) == 6
    for elem in connect:
        n1, n2, n3 = (elem[0], elem[1], elem[2])
        x1, y1 = coords[n1]
        x2, y2 = coords[n2]
        x3, y3 = coords[n3]
        signed_area2 = (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
        assert signed_area2 > 0.0
    for elem in connect:
        n1, n2, n3, n4, n5, n6 = elem
        assert np.allclose(coords[n4], 0.5 * (coords[n1] + coords[n2]))
        assert np.allclose(coords[n5], 0.5 * (coords[n2] + coords[n3]))
        assert np.allclose(coords[n6], 0.5 * (coords[n3] + coords[n1]))
    edge_mid_map = {}
    for elem in connect:
        n1, n2, n3, n4, n5, n6 = elem
        edges = [((n1, n2), n4), ((n2, n3), n5), ((n3, n1), n6)]
        for (a, b), m in edges:
            key = (min(a, b), max(a, b))
            if key in edge_mid_map:
                assert edge_mid_map[key] == m
            else:
                edge_mid_map[key] = m

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
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