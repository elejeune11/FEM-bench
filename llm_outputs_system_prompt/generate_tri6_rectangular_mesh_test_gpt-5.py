def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    expected_nodes = npx * npy
    expected_elements = 2 * nx * ny
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elements, 6)
    assert coords.dtype == coords.astype('float64').dtype
    assert connect.dtype == connect.astype('int64').dtype
    assert coords[0, 0] == xl and coords[0, 1] == yl
    assert coords[npx - 1, 0] == xh and coords[npx - 1, 1] == yl
    assert coords[(npy - 1) * npx, 0] == xl and coords[(npy - 1) * npx, 1] == yh
    assert coords[-1, 0] == xh and coords[-1, 1] == yh
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    expected_xs = [xl + 0.5 * dx * i for i in range(npx)]
    expected_ys = [yl + 0.5 * dy * j for j in range(npy)]
    xs = sorted(set((float(coords[i, 0]) for i in range(coords.shape[0]))))
    ys = sorted(set((float(coords[i, 1]) for i in range(coords.shape[0]))))
    assert xs == expected_xs
    assert ys == expected_ys
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords2.shape == coords.shape
    assert connect2.shape == connect.shape
    assert coords2.dtype == coords.dtype
    assert connect2.dtype == connect.dtype
    assert (coords == coords2).all()
    assert (connect == connect2).all()

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.25, 0.3, 2.75, 3.9)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nn = coords.shape[0]
    for e in range(connect.shape[0]):
        row = connect[e]
        vals = [int(v) for v in row.tolist()]
        assert min(vals) >= 0
        assert max(vals) < nn
        assert len(set(vals)) == 6

    def signed_area(p1, p2, p3):
        (x1, y1) = (p1[0], p1[1])
        (x2, y2) = (p2[0], p2[1])
        (x3, y3) = (p3[0], p3[1])
        return (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1)
    for e in range(connect.shape[0]):
        (n1, n2, n3, n4, n5, n6) = [int(v) for v in connect[e].tolist()]
        p1 = coords[n1]
        p2 = coords[n2]
        p3 = coords[n3]
        area_twice = signed_area(p1, p2, p3)
        assert area_twice > 0.0

        def approx_eq(a, b, tol=1e-12):
            return abs(a - b) <= tol
        pm = coords[n4]
        assert approx_eq(pm[0], 0.5 * (p1[0] + p2[0]))
        assert approx_eq(pm[1], 0.5 * (p1[1] + p2[1]))
        pm = coords[n5]
        assert approx_eq(pm[0], 0.5 * (p2[0] + p3[0]))
        assert approx_eq(pm[1], 0.5 * (p2[1] + p3[1]))
        pm = coords[n6]
        assert approx_eq(pm[0], 0.5 * (p3[0] + p1[0]))
        assert approx_eq(pm[1], 0.5 * (p3[1] + p1[1]))
    edge_mid_map = {}
    edge_count = {}
    for e in range(connect.shape[0]):
        (n1, n2, n3, n4, n5, n6) = [int(v) for v in connect[e].tolist()]
        edges = [(n1, n2, n4), (n2, n3, n5), (n3, n1, n6)]
        for (a, b, m) in edges:
            key = (a, b) if a < b else (b, a)
            edge_mid_map.setdefault(key, []).append(m)
            edge_count[key] = edge_count.get(key, 0) + 1
    for (key, mids) in edge_mid_map.items():
        if edge_count[key] > 1:
            first = mids[0]
            for mid in mids[1:]:
                assert mid == first

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    try:
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    except ValueError:
        pass
    else:
        assert False
    try:
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    except ValueError:
        pass
    else:
        assert False
    try:
        fcn(1.0, 0.0, 1.0, 2.0, 1, 1)
    except ValueError:
        pass
    else:
        assert False
    try:
        fcn(0.0, 2.0, 1.0, 2.0, 1, 1)
    except ValueError:
        pass
    else:
        assert False