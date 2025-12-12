def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    nnodes = npx * npy
    nelems = 2 * nx * ny
    assert coords.shape == (nnodes, 2)
    assert coords.dtype == np.float64
    assert connect.shape == (nelems, 6)
    assert connect.dtype == np.int64
    xs = coords[:, 0]
    ys = coords[:, 1]
    assert np.isclose(xs.min(), xl)
    assert np.isclose(xs.max(), xh)
    assert np.isclose(ys.min(), yl)
    assert np.isclose(ys.max(), yh)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    expected_x = np.linspace(xl, xh, npx)
    expected_y = np.linspace(yl, yh, npy)
    ux = np.unique(xs)
    uy = np.unique(ys)
    assert ux.shape[0] == npx
    assert uy.shape[0] == npy
    assert np.allclose(ux, expected_x)
    assert np.allclose(uy, expected_y)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 2.0, 3.0, 5.5)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < nnodes
    for row in connect:
        assert np.unique(row).size == 6

    def signed_area(tri):
        (n1, n2, n3) = tri[:3]
        (x1, y1) = coords[n1]
        (x2, y2) = coords[n2]
        (x3, y3) = coords[n3]
        return 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
    areas = np.array([signed_area(row) for row in connect])
    assert np.all(areas > 0)
    for row in connect:
        (n1, n2, n3, n4, n5, n6) = row
        (c1, c2, c3) = (coords[n1], coords[n2], coords[n3])
        (c4, c5, c6) = (coords[n4], coords[n5], coords[n6])
        assert np.allclose(c4, 0.5 * (c1 + c2))
        assert np.allclose(c5, 0.5 * (c2 + c3))
        assert np.allclose(c6, 0.5 * (c3 + c1))
    edge_mid_map = {}
    for row in connect:
        (n1, n2, n3, n4, n5, n6) = row
        edges = [(n1, n2, n4), (n2, n3, n5), (n3, n1, n6)]
        for (a, b, m) in edges:
            key = (a, b) if a < b else (b, a)
            if key in edge_mid_map:
                assert edge_mid_map[key] == m
            else:
                edge_mid_map[key] = m

def test_tri6_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs.
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
        fcn(1.0, 0.0, 1.0, 2.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, -1.0, 2.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 2.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 3.0, 2.0, -1.0, 1, 1)