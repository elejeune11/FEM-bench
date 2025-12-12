def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain."""
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords1.shape == ((2 * nx + 1) * (2 * ny + 1), 2)
    assert connect1.shape == (2 * nx * ny, 6)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert np.all(coords1 == coords2)
    assert np.all(connect1 == connect2)
    assert np.all(coords1[:, 0] >= xl) and np.all(coords1[:, 0] <= xh)
    assert np.all(coords1[:, 1] >= yl) and np.all(coords1[:, 1] <= yh)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    for iy in range(2 * ny + 1):
        for ix in range(2 * nx + 1):
            node_id = iy * (2 * nx + 1) + ix
            assert np.allclose(coords1[node_id], [xl + 0.5 * dx * ix, yl + 0.5 * dy * iy])

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain."""
    (xl, yl, xh, yh) = (-1.0, 0.0, 3.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.all(connect >= 0) and np.all(connect < len(coords))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    for elem in connect:
        (N1, N2, N3, N4, N5, N6) = coords[elem]
        assert np.cross(N2 - N1, N3 - N1) > 0
        assert np.allclose(N4, 0.5 * (N1 + N2))
        assert np.allclose(N5, 0.5 * (N2 + N3))
        assert np.allclose(N6, 0.5 * (N3 + N1))
    edge_nodes = {}
    for elem in connect:
        edges = [(elem[0], elem[1]), (elem[1], elem[2]), (elem[2], elem[0])]
        edges = tuple(sorted(edges))
        for edge in edges:
            if edge not in edge_nodes:
                edge_nodes[edge] = set([elem[3] if edge == (elem[0], elem[1]) else elem[4] if edge == (elem[1], elem[2]) else elem[5]])
            else:
                existing_midside = list(edge_nodes[edge])[0]
                if edge == (elem[0], elem[1]):
                    assert existing_midside == elem[3]
                elif edge == (elem[1], elem[2]):
                    assert existing_midside == elem[4]
                else:
                    assert existing_midside == elem[5]

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs."""
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)