def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nnodes = (2 * nx + 1) * (2 * ny + 1)
    expected_nelems = 2 * nx * ny
    assert coords1.shape == (expected_nnodes, 2)
    assert connect1.shape == (expected_nelems, 6)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    assert coords1.shape == coords2.shape
    assert connect1.shape == connect2.shape
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    (dx, dy) = ((xh - xl) / nx, (yh - yl) / ny)
    for iy in range(npy):
        for ix in range(npx):
            node_id = iy * npx + ix
            expected_x = xl + 0.5 * dx * ix
            expected_y = yl + 0.5 * dy * iy
            assert np.isclose(coords1[node_id, 0], expected_x)
            assert np.isclose(coords1[node_id, 1], expected_y)
    corner_nodes = [0, npx - 1, (npy - 1) * npx, (npy - 1) * npx + npx - 1]
    expected_corners = [(xl, yl), (xh, yl), (xl, yh), (xh, yh)]
    for (node_id, (exp_x, exp_y)) in zip(corner_nodes, expected_corners):
        assert np.isclose(coords1[node_id, 0], exp_x)
        assert np.isclose(coords1[node_id, 1], exp_y)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 2.0, 3.0, 5.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    for elem_id in range(connect.shape[0]):
        elem_nodes = connect[elem_id]
        assert len(np.unique(elem_nodes)) == 6
        assert np.all(elem_nodes >= 0)
        assert np.all(elem_nodes < nnodes)
        (n1, n2, n3, n4, n5, n6) = elem_nodes
        (p1, p2, p3) = (coords[n1], coords[n2], coords[n3])
        cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        assert cross_product > 0
        midside_n4 = coords[n4]
        expected_n4 = 0.5 * (coords[n1] + coords[n2])
        assert np.allclose(midside_n4, expected_n4)
        midside_n5 = coords[n5]
        expected_n5 = 0.5 * (coords[n2] + coords[n3])
        assert np.allclose(midside_n5, expected_n5)
        midside_n6 = coords[n6]
        expected_n6 = 0.5 * (coords[n3] + coords[n1])
        assert np.allclose(midside_n6, expected_n6)
    edge_to_midnode = {}
    for elem_id in range(connect.shape[0]):
        (n1, n2, n3, n4, n5, n6) = connect[elem_id]
        edges_and_midnodes = [(tuple(sorted([n1, n2])), n4), (tuple(sorted([n2, n3])), n5), (tuple(sorted([n3, n1])), n6)]
        for (edge, midnode) in edges_and_midnodes:
            if edge in edge_to_midnode:
                assert edge_to_midnode[edge] == midnode
            else:
                edge_to_midnode[edge] = midnode

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
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
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)