def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    xl, yl, xh, yh = (0.0, 0.0, 1.0, 1.0)
    nx, ny = (2, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    npx, npy = (2 * nx + 1, 2 * ny + 1)
    expected_nnodes = npx * npy
    expected_nelems = 2 * nx * ny
    assert coords.shape == (expected_nnodes, 2)
    assert connect.shape == (expected_nelems, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    X = np.linspace(xl, xh, npx, dtype=np.float64)
    Y = np.linspace(yl, yh, npy, dtype=np.float64)
    Xg, Yg = np.meshgrid(X, Y, indexing='xy')
    expected_coords = np.column_stack([Xg.ravel(order='C'), Yg.ravel(order='C')])
    assert np.allclose(coords, expected_coords, atol=0, rtol=0)

    def node_id(ix, iy):
        return iy * npx + ix
    bl = node_id(0, 0)
    br = node_id(npx - 1, 0)
    tl = node_id(0, npy - 1)
    tr = node_id(npx - 1, npy - 1)
    assert np.allclose(coords[bl], [xl, yl])
    assert np.allclose(coords[br], [xh, yl])
    assert np.allclose(coords[tl], [xl, yh])
    assert np.allclose(coords[tr], [xh, yh])
    coords2, connect2 = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    xl, yl, xh, yh = (-1.3, 2.2, 3.7, 6.8)
    nx, ny = (3, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < nnodes
    for row in connect:
        assert len(np.unique(row)) == 6

    def orient(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        return v1[0] * v2[1] - v1[1] * v2[0]
    for e in connect:
        p1, p2, p3 = (coords[e[0]], coords[e[1]], coords[e[2]])
        area2 = orient(p1, p2, p3)
        assert area2 > 0.0
        mid_checks = [(0, 1, 3), (1, 2, 4), (2, 0, 5)]
        for i, j, m in mid_checks:
            expected_mid = 0.5 * (coords[e[i]] + coords[e[j]])
            assert np.allclose(coords[e[m]], expected_mid, atol=1e-12, rtol=0)
    edge_mid_map = {}
    for e in connect:
        edges = [(e[0], e[1], e[3]), (e[1], e[2], e[4]), (e[2], e[0], e[5])]
        for a, b, m in edges:
            key = tuple(sorted((a, b)))
            edge_mid_map.setdefault(key, set()).add(m)
    assert all((len(mids) == 1 for mids in edge_mid_map.values()))

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
        fcn(2.0, 0.0, -1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 2.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 5.0, 1.0, -2.0, 1, 1)