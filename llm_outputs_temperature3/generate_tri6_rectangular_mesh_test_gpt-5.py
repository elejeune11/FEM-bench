def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    n_nodes = npx * npy
    n_elems = 2 * nx * ny
    assert coords.shape == (n_nodes, 2)
    assert connect.shape == (n_elems, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    corners = {(xl, yl), (xh, yl), (xl, yh), (xh, yh)}
    coords_set = {tuple(p) for p in coords}
    for c in corners:
        assert c in coords_set
    expected_x = np.linspace(xl, xh, npx)
    expected_y = np.linspace(yl, yh, npy)
    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    assert np.allclose(unique_x, expected_x)
    assert np.allclose(unique_y, expected_y)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 2.0, 3.0, 4.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < n_nodes
    for elem in connect:
        assert len(set(elem.tolist())) == 6

    def is_ccw(tri):
        (p1, p2, p3) = (coords[tri[0]], coords[tri[1]], coords[tri[2]])
        area2 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        return area2 > 0
    for elem in connect:
        (N1, N2, N3, N4, N5, N6) = elem
        assert is_ccw([N1, N2, N3])
        (p1, p2, p3) = (coords[N1], coords[N2], coords[N3])
        (p4, p5, p6) = (coords[N4], coords[N5], coords[N6])
        assert np.allclose(p4, 0.5 * (p1 + p2))
        assert np.allclose(p5, 0.5 * (p2 + p3))
        assert np.allclose(p6, 0.5 * (p3 + p1))
    edge_to_midside = {}
    edge_count = {}
    for elem in connect:
        (N1, N2, N3, N4, N5, N6) = elem
        edges = [(N1, N2, N4), (N2, N3, N5), (N3, N1, N6)]
        for (a, b, m) in edges:
            key = tuple(sorted((a, b)))
            edge_to_midside.setdefault(key, set()).add(m)
            edge_count[key] = edge_count.get(key, 0) + 1
    for mids in edge_to_midside.values():
        assert len(mids) == 1
    counts = list(edge_count.values())
    assert max(counts) == 2
    assert min(counts) == 1
    assert any((c == 2 for c in counts))
    assert any((c == 1 for c in counts))

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