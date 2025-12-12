def test_tri6_mesh_basic_structure_and_determinism(fcn):
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    assert coords.shape == (npx * npy, 2)
    assert connect.shape == (2 * nx * ny, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    for iy in range(npy):
        for ix in range(npx):
            node_id = iy * npx + ix
            assert np.isclose(coords[node_id, 0], xl + 0.5 * dx * ix)
            assert np.isclose(coords[node_id, 1], yl + 0.5 * dy * iy)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)
    assert np.isclose(coords[0, 0], xl)
    assert np.isclose(coords[0, 1], yl)
    assert np.isclose(coords[npx - 1, 0], xh)
    assert np.isclose(coords[npx - 1, 1], yl)
    assert np.isclose(coords[(npy - 1) * npx, 0], xl)
    assert np.isclose(coords[(npy - 1) * npx, 1], yh)
    assert np.isclose(coords[-1, 0], xh)
    assert np.isclose(coords[-1, 1], yh)

def test_tri6_mesh_geometry_and_conformity(fcn):
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (1, 1)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    assert len(connect) == 2
    assert np.all(connect >= 0) and np.all(connect < npx * npy)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    tri1 = connect[0]
    (n1, n2, n3) = (tri1[0], tri1[1], tri1[2])
    (p1, p2, p3) = (coords[n1], coords[n2], coords[n3])
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    assert cross > 0
    (n4, n5, n6) = (tri1[3], tri1[4], tri1[5])
    assert np.allclose(coords[n4], (p1 + p2) / 2)
    assert np.allclose(coords[n5], (p2 + p3) / 2)
    assert np.allclose(coords[n6], (p3 + p1) / 2)
    tri2 = connect[1]
    (n1, n2, n3) = (tri2[0], tri2[1], tri2[2])
    (p1, p2, p3) = (coords[n1], coords[n2], coords[n3])
    cross = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
    assert cross > 0
    (n4, n5, n6) = (tri2[3], tri2[4], tri2[5])
    assert np.allclose(coords[n4], (p1 + p2) / 2)
    assert np.allclose(coords[n5], (p2 + p3) / 2)
    assert np.allclose(coords[n6], (p3 + p1) / 2)
    assert coords[tri1[3]] == coords[tri2[5]]
    assert tri1[0] == tri2[2]
    assert tri1[1] == tri2[1]
    assert len(set(tri1) | set(tri2)) == 10
    assert len(set(tri1) | set(tri2)) == 9

def test_tri6_mesh_invalid_inputs(fcn):
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, -1)