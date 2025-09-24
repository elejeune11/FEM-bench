def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    expected_nodes = npx * npy
    expected_elems = 2 * nx * ny
    assert coords.shape == (expected_nodes, 2)
    assert coords.dtype == np.float64
    assert connect.shape == (expected_elems, 6)
    assert connect.dtype == np.int64
    xs = np.unique(coords[:, 0])
    ys = np.unique(coords[:, 1])
    assert len(xs) == npx
    assert len(ys) == npy
    assert np.isclose(xs[0], xl) and np.isclose(xs[-1], xh)
    assert np.isclose(ys[0], yl) and np.isclose(ys[-1], yh)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    step_x = dx / 2.0
    step_y = dy / 2.0
    if len(xs) > 1:
        assert np.allclose(np.diff(xs), step_x)
    if len(ys) > 1:
        assert np.allclose(np.diff(ys), step_y)

    def node_id(ix, iy):
        return iy * npx + ix
    corner_ids = [node_id(0, 0), node_id(npx - 1, 0), node_id(npx - 1, npy - 1), node_id(0, npy - 1)]
    expected_corner_coords = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]], dtype=np.float64)
    assert np.allclose(coords[corner_ids], expected_corner_coords)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-2.5, 1.0, 3.0, 4.5)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    N = coords.shape[0]
    assert connect.ndim == 2 and connect.shape[1] == 6
    assert connect.min() >= 0
    assert connect.max() < N
    for row in connect:
        assert len(np.unique(row)) == 6
    for row in connect:
        (N1, N2, N3, N4, N5, N6) = row
        (p1, p2, p3) = (coords[N1], coords[N2], coords[N3])
        orient = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p3[0] - p1[0]) * (p2[1] - p1[1])
        assert orient > 0
        assert np.allclose(coords[N4], 0.5 * (p1 + p2))
        assert np.allclose(coords[N5], 0.5 * (p2 + p3))
        assert np.allclose(coords[N6], 0.5 * (p3 + p1))
    edge_to_mid = {}
    for row in connect:
        (N1, N2, N3, N4, N5, N6) = row
        edges = [(tuple(sorted((N1, N2))), N4), (tuple(sorted((N2, N3))), N5), (tuple(sorted((N3, N1))), N6)]
        for (edge, mid) in edges:
            if edge in edge_to_mid:
                assert edge_to_mid[edge] == mid
            else:
                edge_to_mid[edge] = mid

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
        fcn(0.0, 0.0, 1.0, 1.0, 1, -1)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 2.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 2.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 2.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 2.0, 1.0, 1, 1)