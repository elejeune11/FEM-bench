def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    xl, yl, xh, yh = (0.0, 0.0, 1.0, 1.0)
    nx, ny = (2, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    npx, npy = (2 * nx + 1, 2 * ny + 1)
    expected_nodes = npx * npy - nx * ny
    expected_elements = nx * ny
    assert coords.shape == (expected_nodes, 2)
    assert coords.dtype == np.float64
    assert connect.shape == (expected_elements, 8)
    assert connect.dtype == np.int64
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]], dtype=np.float64)
    for c in corners:
        assert np.any(np.all(np.isclose(coords, c, rtol=0, atol=1e-14), axis=1))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    hx, hy = (0.5 * dx, 0.5 * dy)
    ux = np.unique(coords[:, 0])
    uy = np.unique(coords[:, 1])
    assert len(ux) == 2 * nx + 1
    assert len(uy) == 2 * ny + 1
    if len(ux) > 1:
        assert np.allclose(np.diff(ux), hx)
    if len(uy) > 1:
        assert np.allclose(np.diff(uy), hy)
    centers = []
    for cx in range(nx):
        for cy in range(ny):
            centers.append([xl + (cx + 0.5) * dx, yl + (cy + 0.5) * dy])
    centers = np.array(centers, dtype=np.float64)
    for ctr in centers:
        assert not np.any(np.all(np.isclose(coords, ctr, rtol=0, atol=1e-14), axis=1))
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
    assert connect.dtype == np.int64
    assert np.all(connect >= 0) and np.all(connect < Nnodes)
    for e in range(Ne):
        row = connect[e]
        assert len(np.unique(row)) == 8
    for e in range(Ne):
        N1, N2, N3, N4, N5, N6, N7, N8 = connect[e]
        p1, p2, p3, p4 = (coords[N1], coords[N2], coords[N3], coords[N4])
        m5, m6, m7, m8 = (coords[N5], coords[N6], coords[N7], coords[N8])
        area = 0.5 * (p1[0] * p2[1] + p2[0] * p3[1] + p3[0] * p4[1] + p4[0] * p1[1])
        assert area > 0.0
        assert np.allclose(m5, 0.5 * (p1 + p2))
        assert np.allclose(m6, 0.5 * (p2 + p3))
        assert np.allclose(m7, 0.5 * (p3 + p4))
        assert np.allclose(m8, 0.5 * (p4 + p1))
    edge_counts = {}
    for e in range(Ne):
        N1, N2, N3, N4 = (connect[e, 0], connect[e, 1], connect[e, 2], connect[e, 3])
        edges = [(N1, N2), (N2, N3), (N3, N4), (N4, N1)]
        for a, b in edges:
            key = (a, b) if a < b else (b, a)
            edge_counts[key] = edge_counts.get(key, 0) + 1
    counts = np.array(list(edge_counts.values()))
    assert np.all(np.isin(counts, [1, 2]))
    expected_interior = (nx - 1) * ny + nx * (ny - 1)
    num_interior = int(np.sum(counts == 2))
    assert num_interior == expected_interior

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
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