def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    xl, yl, xh, yh = (0.0, 0.0, 1.0, 1.0)
    nx, ny = (2, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    n_nodes = npx * npy
    n_elems = 2 * nx * ny
    assert isinstance(coords, np.ndarray)
    assert isinstance(connect, np.ndarray)
    assert coords.shape == (n_nodes, 2)
    assert connect.shape == (n_elems, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64

    def node_id(ix, iy):
        return iy * npx + ix
    assert np.allclose(coords[node_id(0, 0)], [xl, yl])
    assert np.allclose(coords[node_id(npx - 1, 0)], [xh, yl])
    assert np.allclose(coords[node_id(0, npy - 1)], [xl, yh])
    assert np.allclose(coords[node_id(npx - 1, npy - 1)], [xh, yh])
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    hx = 0.5 * dx
    hy = 0.5 * dy
    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    assert len(unique_x) == npx and len(unique_y) == npy
    assert np.allclose(unique_x, xl + hx * np.arange(npx))
    assert np.allclose(unique_y, yl + hy * np.arange(npy))
    coords2, connect2 = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    xl, xh = (-1.0, 2.0)
    yl, yh = (1.5, 3.0)
    nx, ny = (3, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    for row in connect:
        assert row.dtype == np.int64
        assert np.all(row >= 0)
        assert np.all(row < n_nodes)
        assert len(set(row.tolist())) == 6

    def area2(p1, p2, p3):
        v1 = p2 - p1
        v2 = p3 - p1
        return v1[0] * v2[1] - v1[1] * v2[0]
    for tri in connect:
        n1, n2, n3 = (tri[0], tri[1], tri[2])
        p1, p2, p3 = (coords[n1], coords[n2], coords[n3])
        a2 = area2(p1, p2, p3)
        assert a2 > 0.0
    edge_mid = {}
    for tri in connect:
        n1, n2, n3, n4, n5, n6 = tri.tolist()
        p1, p2, p3 = (coords[n1], coords[n2], coords[n3])
        pm4, pm5, pm6 = (coords[n4], coords[n5], coords[n6])
        assert np.allclose(pm4, 0.5 * (p1 + p2))
        assert np.allclose(pm5, 0.5 * (p2 + p3))
        assert np.allclose(pm6, 0.5 * (p3 + p1))
        for (a, b), m in [((n1, n2), n4), ((n2, n3), n5), ((n3, n1), n6)]:
            key = (a, b) if a < b else (b, a)
            if key in edge_mid:
                assert edge_mid[key] == m
            else:
                edge_mid[key] = m

def test_tri6_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -1, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, -3)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 0.0, 1, 1)