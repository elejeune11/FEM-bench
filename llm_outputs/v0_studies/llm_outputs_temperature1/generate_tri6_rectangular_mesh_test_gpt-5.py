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
    n_nodes = npx * npy
    n_elem = 2 * nx * ny
    assert isinstance(coords, np.ndarray) and isinstance(connect, np.ndarray)
    assert coords.shape == (n_nodes, 2)
    assert connect.shape == (n_elem, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64

    def nid(ix, iy):
        return iy * npx + ix
    corners = {nid(0, 0): (xl, yl), nid(2 * nx, 0): (xh, yl), nid(0, 2 * ny): (xl, yh), nid(2 * nx, 2 * ny): (xh, yh)}
    for (idx, (ex, ey)) in corners.items():
        assert np.allclose(coords[idx], [ex, ey])
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    (IX, IY) = np.meshgrid(np.arange(npx), np.arange(npy), indexing='xy')
    expected_coords = np.column_stack([xl + 0.5 * dx * IX.ravel(order='C'), yl + 0.5 * dy * IY.ravel(order='C')])
    assert np.allclose(coords, expected_coords)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(connect, connect2)
    assert np.array_equal(coords, coords2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-2.5, 1.0, 3.5, 4.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    n_nodes = npx * npy
    assert connect.min() >= 0
    assert connect.max() < n_nodes
    for row in connect:
        assert len(set(row.tolist())) == 6

    def signed_area2(p1, p2, p3):
        (x1, y1) = p1
        (x2, y2) = p2
        (x3, y3) = p3
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    for e in connect:
        (n1, n2, n3, n4, n5, n6) = e
        (p1, p2, p3) = (coords[n1], coords[n2], coords[n3])
        (p4, p5, p6) = (coords[n4], coords[n5], coords[n6])
        area2 = signed_area2(p1, p2, p3)
        assert area2 > 0.0
        assert np.allclose(p4, 0.5 * (p1 + p2))
        assert np.allclose(p5, 0.5 * (p2 + p3))
        assert np.allclose(p6, 0.5 * (p3 + p1))
    mid_by_edge = {}
    counts = {}
    for e in connect:
        (n1, n2, n3, n4, n5, n6) = e
        edges = [((n1, n2), n4), ((n2, n3), n5), ((n3, n1), n6)]
        for ((a, b), mid) in edges:
            key = (a, b) if a < b else (b, a)
            if key in mid_by_edge:
                assert mid_by_edge[key] == mid
            else:
                mid_by_edge[key] = mid
            counts[key] = counts.get(key, 0) + 1
    for c in counts.values():
        assert c in (1, 2)

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
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)