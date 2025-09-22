def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl) = (0.0, 0.0)
    (xh, yh) = (2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    assert coords.shape == (npx * npy, 2)
    assert connect.shape == (2 * nx * ny, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.allclose(coords[0], [xl, yl])
    assert np.allclose(coords[npx - 1], [xh, yl])
    assert np.allclose(coords[npx * (npy - 1)], [xl, yh])
    assert np.allclose(coords[-1], [xh, yh])
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    for ix in range(npx):
        for iy in range(npy):
            node_id = iy * npx + ix
            assert np.allclose(coords[node_id], [xl + 0.5 * dx * ix, yl + 0.5 * dy * iy])
    assert np.all(coords == coords2)
    assert np.all(connect == connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl) = (0.0, 1.0)
    (xh, yh) = (3.0, 5.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    num_nodes = npx * npy
    for element in connect:
        assert np.all(0 <= element) and np.all(element < num_nodes)
        assert len(np.unique(element)) == 6
        (n1, n2, n3, n4, n5, n6) = element
        v1 = coords[n2] - coords[n1]
        v2 = coords[n3] - coords[n1]
        assert np.cross(v1, v2) > 0
        assert np.allclose(coords[n4], 0.5 * (coords[n1] + coords[n2]))
        assert np.allclose(coords[n5], 0.5 * (coords[n2] + coords[n3]))
        assert np.allclose(coords[n6], 0.5 * (coords[n3] + coords[n1]))
    assert connect[0, 1] == connect[1, 1]
    assert connect[0, 2] == connect[1, 2]