def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    xl, yl, xh, yh = (0.0, 0.0, 1.0, 1.0)
    nx = ny = 2
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    Nnodes = npx * npy
    Ne = 2 * nx * ny
    assert isinstance(coords, np.ndarray) and isinstance(connect, np.ndarray)
    assert coords.shape == (Nnodes, 2)
    assert connect.shape == (Ne, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    xs = coords[:, 0]
    ys = coords[:, 1]
    assert np.isclose(xs.min(), xl)
    assert np.isclose(xs.max(), xh)
    assert np.isclose(ys.min(), yl)
    assert np.isclose(ys.max(), yh)
    id_bl = 0
    id_br = npx - 1
    id_tl = (npy - 1) * npx
    id_tr = (npy - 1) * npx + (npx - 1)
    assert np.allclose(coords[id_bl], [xl, yl])
    assert np.allclose(coords[id_br], [xh, yl])
    assert np.allclose(coords[id_tl], [xl, yh])
    assert np.allclose(coords[id_tr], [xh, yh])
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    expected_x = np.array([xl + 0.5 * dx * ix for ix in range(npx)], dtype=np.float64)
    expected_y = np.array([yl + 0.5 * dy * iy for iy in range(npy)], dtype=np.float64)
    unique_x = np.unique(xs)
    unique_y = np.unique(ys)
    assert unique_x.size == npx
    assert unique_y.size == npy
    assert np.allclose(unique_x, expected_x, rtol=0.0, atol=1e-15)
    assert np.allclose(unique_y, expected_y, rtol=0.0, atol=1e-15)
    coords2, connect2 = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    xl, yl, xh, yh = (1.0, -2.0, 4.0, 2.0)
    nx, ny = (3, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < Nnodes
    assert all((len(np.unique(conn)) == 6 for conn in connect))
    p1 = coords[connect[:, 0]]
    p2 = coords[connect[:, 1]]
    p3 = coords[connect[:, 2]]
    cross = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p2[:, 1] - p1[:, 1]) * (p3[:, 0] - p1[:, 0])
    assert np.all(cross > 0.0)
    mid12 = coords[connect[:, 3]]
    mid23 = coords[connect[:, 4]]
    mid31 = coords[connect[:, 5]]
    assert np.allclose(mid12, 0.5 * (p1 + p2), rtol=0.0, atol=1e-14)
    assert np.allclose(mid23, 0.5 * (p2 + p3), rtol=0.0, atol=1e-14)
    assert np.allclose(mid31, 0.5 * (p3 + p1), rtol=0.0, atol=1e-14)
    edge_mid_map = {}
    edge_count = {}
    for conn in connect:
        edges = ((conn[0], conn[1], conn[3]), (conn[1], conn[2], conn[4]), (conn[2], conn[0], conn[5]))
        for a, b, m in edges:
            key = (a, b) if a < b else (b, a)
            prev = edge_mid_map.get(key)
            if prev is None:
                edge_mid_map[key] = m
            else:
                assert prev == m
            edge_count[key] = edge_count.get(key, 0) + 1
    assert all((c in (1, 2) for c in edge_count.values()))

def test_tri6_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, -2)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 2.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 2.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 2.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 2.0, 1.0, 1, 1)