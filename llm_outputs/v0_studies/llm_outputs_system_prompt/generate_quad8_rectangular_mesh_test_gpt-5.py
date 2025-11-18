def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    expected_nodes = npx * npy - nx * ny
    expected_elements = nx * ny
    assert coords.ndim == 2 and coords.shape[1] == 2
    assert coords.shape[0] == expected_nodes
    assert connect.ndim == 2 and connect.shape == (expected_elements, 8)
    assert str(coords.dtype) == 'float64'
    assert str(connect.dtype) == 'int64'
    coord_set = set(map(tuple, coords.tolist()))
    corners = [(xl, yl), (xh, yl), (xh, yh), (xl, yh)]
    for c in corners:
        assert c in coord_set
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    step_x = 0.5 * dx
    step_y = 0.5 * dy
    expected_xs = [xl + i * step_x for i in range(npx)]
    expected_ys = [yl + j * step_y for j in range(npy)]
    xs = sorted(set([p[0] for p in coords.tolist()]))
    ys = sorted(set([p[1] for p in coords.tolist()]))
    assert xs == expected_xs
    assert ys == expected_ys
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords.shape == coords2.shape
    assert connect.shape == connect2.shape
    assert (coords == coords2).all()
    assert (connect == connect2).all()

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nn = coords.shape[0]
    assert int(connect.min()) >= 0
    assert int(connect.max()) < nn
    for e in connect:
        assert len(set(e.tolist())) == 8
    for e in connect.tolist():
        (N1, N2, N3, N4, N5, N6, N7, N8) = e
        (x1, y1) = coords[N1]
        (x2, y2) = coords[N2]
        (x3, y3) = coords[N3]
        (x4, y4) = coords[N4]
        (xb, yb) = coords[N5]
        (xr, yr) = coords[N6]
        (xt, yt) = coords[N7]
        (xlft, ylft) = coords[N8]
        area2 = x1 * y2 - x2 * y1 + (x2 * y3 - x3 * y2) + (x3 * y4 - x4 * y3) + (x4 * y1 - x1 * y4)
        assert area2 > 0.0
        assert xb == (x1 + x2) / 2.0 and yb == (y1 + y2) / 2.0
        assert xr == (x2 + x3) / 2.0 and yr == (y2 + y3) / 2.0
        assert xt == (x3 + x4) / 2.0 and yt == (y3 + y4) / 2.0
        assert xlft == (x4 + x1) / 2.0 and ylft == (y4 + y1) / 2.0
    edge_counts = {}
    for e in connect.tolist():
        (N1, N2, N3, N4) = (e[0], e[1], e[2], e[3])
        edges = [(N1, N2), (N2, N3), (N3, N4), (N4, N1)]
        for (a, b) in edges:
            key = frozenset((int(a), int(b)))
            edge_counts[key] = edge_counts.get(key, 0) + 1
    unique_edges = len(edge_counts)
    expected_unique_edges = nx * (ny + 1) + ny * (nx + 1)
    assert unique_edges == expected_unique_edges
    counts = list(edge_counts.values())
    num_once = counts.count(1)
    num_twice = counts.count(2)
    expected_boundary_edges = 2 * (nx + ny)
    expected_internal_edges = (nx - 1) * ny + nx * (ny - 1)
    assert num_once == expected_boundary_edges
    assert num_twice == expected_internal_edges
    assert all((c in (1, 2) for c in counts))

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    raised = False
    try:
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    except ValueError:
        raised = True
    assert raised
    raised = False
    try:
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    except ValueError:
        raised = True
    assert raised
    raised = False
    try:
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    except ValueError:
        raised = True
    assert raised
    raised = False
    try:
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    except ValueError:
        raised = True
    assert raised