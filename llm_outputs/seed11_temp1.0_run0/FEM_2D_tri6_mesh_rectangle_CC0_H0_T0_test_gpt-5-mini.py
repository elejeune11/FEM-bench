def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    Nnodes = npx * npy
    Ne = 2 * nx * ny
    assert coords.shape == (Nnodes, 2)
    assert connect.shape == (Ne, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    id_bl = 0
    id_br = npx - 1
    id_tl = (npy - 1) * npx
    id_tr = Nnodes - 1
    assert np.allclose(coords[id_bl], [xl, yl])
    assert np.allclose(coords[id_br], [xh, yl])
    assert np.allclose(coords[id_tl], [xl, yh])
    assert np.allclose(coords[id_tr], [xh, yh])
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    step_x = 0.5 * dx
    step_y = 0.5 * dy
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    expected_xs = np.array([xl + step_x * ix for ix in range(npx)], dtype=np.float64)
    expected_ys = np.array([yl + step_y * iy for iy in range(npy)], dtype=np.float64)
    assert xs.shape[0] == npx
    assert ys.shape[0] == npy
    assert np.allclose(xs, expected_xs)
    assert np.allclose(ys, expected_ys)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 0.0, 2.0, 4.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    Ne = connect.shape[0]
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    for row in connect:
        assert np.all(row >= 0) and np.all(row < Nnodes)
        assert len(np.unique(row)) == 6
    for row in connect:
        (n1, n2, n3, n4, n5, n6) = row
        p1 = coords[n1]
        p2 = coords[n2]
        p3 = coords[n3]
        area2 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        assert area2 > 0.0
        assert np.allclose(coords[n4], 0.5 * (p1 + p2))
        assert np.allclose(coords[n5], 0.5 * (p2 + p3))
        assert np.allclose(coords[n6], 0.5 * (p3 + p1))
    edge_counts = {}
    for row in connect:
        (n1, n2, n3, n4, n5, n6) = row
        edges = [frozenset((int(n1), int(n2), int(n4))), frozenset((int(n2), int(n3), int(n5))), frozenset((int(n3), int(n1), int(n6)))]
        for e in edges:
            edge_counts[e] = edge_counts.get(e, 0) + 1
    for ecount in edge_counts.values():
        assert ecount in (1, 2)
    shared_edges = sum((1 for v in edge_counts.values() if v == 2))
    assert shared_edges >= 1

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 0, 1)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, yl, 1.0, yh, 1, 1)
    with pytest.raises(ValueError):
        fcn(xl, 1.0, xh, 1.0, 1, 1)