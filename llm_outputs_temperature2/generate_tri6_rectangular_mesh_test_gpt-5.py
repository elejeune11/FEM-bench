def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    Nnodes = npx * npy
    Ne = 2 * nx * ny
    assert coords.shape == (Nnodes, 2)
    assert connect.shape == (Ne, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    corners = np.array([[xl, yl], [xh, yl], [xl, yh], [xh, yh]])
    for c in corners:
        assert np.any(np.all(np.isclose(coords, c), axis=1))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    hx = dx / 2.0
    hy = dy / 2.0
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    assert np.isclose(xs[0], xl) and np.isclose(xs[-1], xh)
    assert np.isclose(ys[0], yl) and np.isclose(ys[-1], yh)
    if len(xs) > 1:
        assert np.allclose(np.diff(xs), hx)
    if len(ys) > 1:
        assert np.allclose(np.diff(ys), hy)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.5, 0.25, 2.0, 3.25)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = (2 * nx + 1) * (2 * ny + 1)
    assert connect.min() >= 0
    assert connect.max() < Nnodes
    for elem in connect:
        assert len(set(elem.tolist())) == 6
    for elem in connect:
        (N1, N2, N3, N4, N5, N6) = elem
        (p1, p2, p3) = (coords[N1], coords[N2], coords[N3])
        (p4, p5, p6) = (coords[N4], coords[N5], coords[N6])
        cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        assert cross > 0.0
        assert np.allclose(p4, 0.5 * (p1 + p2))
        assert np.allclose(p5, 0.5 * (p2 + p3))
        assert np.allclose(p6, 0.5 * (p3 + p1))
    edge_mid = {}
    for elem in connect:
        (N1, N2, N3, N4, N5, N6) = elem
        edges = [((N1, N2), N4), ((N2, N3), N5), ((N3, N1), N6)]
        for ((a, b), m) in edges:
            key = tuple(sorted((a, b)))
            if key in edge_mid:
                assert edge_mid[key] == m
            else:
                edge_mid[key] = m

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -3, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, -1)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)