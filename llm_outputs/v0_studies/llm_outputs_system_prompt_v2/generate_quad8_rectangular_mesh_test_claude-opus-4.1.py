def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nnodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_nelems = nx * ny
    assert coords.shape == (expected_nnodes, 2)
    assert connect.shape == (expected_nelems, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    corner_coords = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]])
    for corner in corner_coords:
        assert np.any(np.all(np.abs(coords - corner) < 1e-14, axis=1))
    for i in range(coords.shape[0]):
        (x, y) = coords[i]
        ix = np.round((x - xl) / (0.5 * dx))
        iy = np.round((y - yl) / (0.5 * dy))
        assert np.abs(x - (xl + 0.5 * dx * ix)) < 1e-14
        assert np.abs(y - (yl + 0.5 * dy * iy)) < 1e-14
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, -2.0, 3.0, 1.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    nelems = connect.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < nnodes)
    for e in range(nelems):
        elem_nodes = connect[e]
        assert len(np.unique(elem_nodes)) == 8
        (n1, n2, n3, n4, n5, n6, n7, n8) = elem_nodes
        p1 = coords[n1]
        p2 = coords[n2]
        p3 = coords[n3]
        p4 = coords[n4]
        area = 0.5 * ((p2[0] - p1[0]) * (p4[1] - p1[1]) - (p4[0] - p1[0]) * (p2[1] - p1[1]))
        area += 0.5 * ((p3[0] - p2[0]) * (p4[1] - p2[1]) - (p4[0] - p2[0]) * (p3[1] - p2[1]))
        assert area > 0
        assert np.allclose(coords[n5], 0.5 * (coords[n1] + coords[n2]))
        assert np.allclose(coords[n6], 0.5 * (coords[n2] + coords[n3]))
        assert np.allclose(coords[n7], 0.5 * (coords[n3] + coords[n4]))
        assert np.allclose(coords[n8], 0.5 * (coords[n4] + coords[n1]))
    edges = {}
    for e in range(nelems):
        elem_nodes = connect[e]
        edge_pairs = [(elem_nodes[0], elem_nodes[1]), (elem_nodes[1], elem_nodes[2]), (elem_nodes[2], elem_nodes[3]), (elem_nodes[3], elem_nodes[0])]
        for (n1, n2) in edge_pairs:
            edge_key = tuple(sorted([n1, n2]))
            if edge_key in edges:
                edges[edge_key] += 1
            else:
                edges[edge_key] = 1
    for (edge_key, count) in edges.items():
        assert count <= 2

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -1, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, -1)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 2, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 2, 2)