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
    nnodes = npx * npy
    nelems = 2 * nx * ny
    assert coords.shape == (nnodes, 2)
    assert connect.shape == (nelems, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.isclose(coords[0, 0], xl) and np.isclose(coords[0, 1], yl)
    assert np.isclose(coords[npx - 1, 0], xh) and np.isclose(coords[npx - 1, 1], yl)
    top_left_id = (npy - 1) * npx + 0
    top_right_id = (npy - 1) * npx + (npx - 1)
    assert np.isclose(coords[top_left_id, 0], xl) and np.isclose(coords[top_left_id, 1], yh)
    assert np.isclose(coords[top_right_id, 0], xh) and np.isclose(coords[top_right_id, 1], yh)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    hx = 0.5 * dx
    hy = 0.5 * dy
    expected_x = xl + hx * np.arange(npx, dtype=np.float64)
    expected_y = yl + hy * np.arange(npy, dtype=np.float64)
    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    assert np.allclose(unique_x, expected_x)
    assert np.allclose(unique_y, expected_y)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.5, 2.0, 2.5, 5.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < nnodes
    for row in connect:
        assert len(set(row.tolist())) == 6
    for row in connect:
        (N1, N2, N3, N4, N5, N6) = row
        (p1, p2, p3) = (coords[N1], coords[N2], coords[N3])
        area2 = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        assert area2 > 0.0
        mid12 = 0.5 * (coords[N1] + coords[N2])
        mid23 = 0.5 * (coords[N2] + coords[N3])
        mid31 = 0.5 * (coords[N3] + coords[N1])
        assert np.allclose(coords[N4], mid12, rtol=0.0, atol=1e-12)
        assert np.allclose(coords[N5], mid23, rtol=0.0, atol=1e-12)
        assert np.allclose(coords[N6], mid31, rtol=0.0, atol=1e-12)
    edge_mid_map = {}
    for row in connect:
        (N1, N2, N3, N4, N5, N6) = row
        edges = [(tuple(sorted((N1, N2))), N4), (tuple(sorted((N2, N3))), N5), (tuple(sorted((N3, N1))), N6)]
        for (key, mid) in edges:
            if key in edge_mid_map:
                assert edge_mid_map[key] == mid
            else:
                edge_mid_map[key] = mid

def test_tri6_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)