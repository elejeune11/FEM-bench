def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords.shape == ((2 * nx + 1) * (2 * ny + 1), 2)
    assert connect.shape == (2 * nx * ny, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.allclose(coords[0], [xl, yl])
    assert np.allclose(coords[2 * nx], [xh, yl])
    assert np.allclose(coords[2 * ny * (2 * nx + 1)], [xl, yh])
    assert np.allclose(coords[-1], [xh, yh])
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    for ix in range(2 * nx + 1):
        for iy in range(2 * ny + 1):
            node_id = iy * (2 * nx + 1) + ix
            expected_coord = [xl + 0.5 * dx * ix, yl + 0.5 * dy * iy]
            assert np.allclose(coords[node_id], expected_coord)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (2, 1)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    max_node_id = npx * npy - 1
    for elem in connect:
        assert np.all(elem >= 0) and np.all(elem <= max_node_id)
        assert len(set(elem)) == 6
    for elem in connect:
        (N1, N2, N3) = elem[:3]
        (x1, y1) = coords[N1]
        (x2, y2) = coords[N2]
        (x3, y3) = coords[N3]
        assert (x2 - x1) * (y3 - y1) - (y2 - y1) * (x3 - x1) > 0
    for elem in connect:
        (N1, N2, N3, N4, N5, N6) = elem
        assert np.allclose(coords[N4], (coords[N1] + coords[N2]) / 2)
        assert np.allclose(coords[N5], (coords[N2] + coords[N3]) / 2)
        assert np.allclose(coords[N6], (coords[N3] + coords[N1]) / 2)
    edge_to_nodes = {}
    for elem in connect:
        for (i, (n1, n2)) in enumerate([(0, 1), (1, 2), (2, 0)]):
            edge = tuple(sorted((elem[n1], elem[n2])))
            if edge in edge_to_nodes:
                assert edge_to_nodes[edge] == elem[3 + i]
            else:
                edge_to_nodes[edge] = elem[3 + i]