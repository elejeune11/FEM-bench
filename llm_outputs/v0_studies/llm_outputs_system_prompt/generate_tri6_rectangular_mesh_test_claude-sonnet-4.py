def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    expected_nnodes = npx * npy
    expected_nelems = 2 * nx * ny
    assert coords1.shape == (expected_nnodes, 2)
    assert connect1.shape == (expected_nelems, 6)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    x_coords = coords1[:, 0]
    y_coords = coords1[:, 1]
    assert np.min(x_coords) == xl
    assert np.max(x_coords) == xh
    assert np.min(y_coords) == yl
    assert np.max(y_coords) == yh
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    expected_x = np.array([xl + 0.5 * dx * ix for ix in range(npx)])
    expected_y = np.array([yl + 0.5 * dy * iy for iy in range(npy)])
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    assert np.allclose(unique_x, expected_x)
    assert np.allclose(unique_y, expected_y)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 2.0, 3.0, 5.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    nelems = connect.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < nnodes)
    for elem in connect:
        assert len(np.unique(elem)) == 6
    for elem in connect:
        (n1, n2, n3, n4, n5, n6) = elem
        (p1, p2, p3) = (coords[n1], coords[n2], coords[n3])
        (p4, p5, p6) = (coords[n4], coords[n5], coords[n6])
        v1 = p2 - p1
        v2 = p3 - p1
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross_z > 0
        assert np.allclose(p4, 0.5 * (p1 + p2))
        assert np.allclose(p5, 0.5 * (p2 + p3))
        assert np.allclose(p6, 0.5 * (p3 + p1))
    edge_to_nodes = {}
    for (elem_idx, elem) in enumerate(connect):
        (n1, n2, n3, n4, n5, n6) = elem
        edges = [(tuple(sorted([n1, n2])), n4), (tuple(sorted([n2, n3])), n5), (tuple(sorted([n3, n1])), n6)]
        for (edge, midnode) in edges:
            if edge in edge_to_nodes:
                assert edge_to_nodes[edge] == midnode
            else:
                edge_to_nodes[edge] = midnode

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 0, ny)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, -1, ny)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, nx, 0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, nx, -1)
    with pytest.raises(ValueError):
        fcn(1.0, yl, 1.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(2.0, yl, 1.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 1.0, xh, 1.0, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 2.0, xh, 1.0, nx, ny)