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
    expected_nnodes = (2 * nx + 1) * (2 * ny + 1)
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
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    for iy in range(npy):
        for ix in range(npx):
            node_id = iy * npx + ix
            expected_x = xl + 0.5 * dx * ix
            expected_y = yl + 0.5 * dy * iy
            assert np.isclose(coords1[node_id, 0], expected_x)
            assert np.isclose(coords1[node_id, 1], expected_y)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (1.0, 2.0, 4.0, 5.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    for elem_idx in range(connect.shape[0]):
        elem_nodes = connect[elem_idx, :]
        assert np.all(elem_nodes >= 0)
        assert np.all(elem_nodes < nnodes)
        assert len(np.unique(elem_nodes)) == 6
    for elem_idx in range(connect.shape[0]):
        (n1, n2, n3, n4, n5, n6) = connect[elem_idx, :]
        (p1, p2, p3) = (coords[n1], coords[n2], coords[n3])
        cross_product = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0] - p1[0])
        assert cross_product > 0
        assert np.allclose(coords[n4], 0.5 * (coords[n1] + coords[n2]))
        assert np.allclose(coords[n5], 0.5 * (coords[n2] + coords[n3]))
        assert np.allclose(coords[n6], 0.5 * (coords[n3] + coords[n1]))
    edge_to_elements = {}
    for elem_idx in range(connect.shape[0]):
        (n1, n2, n3, n4, n5, n6) = connect[elem_idx, :]
        edges = [(n1, n2, n4), (n2, n3, n5), (n3, n1, n6)]
        for edge in edges:
            (v1, v2, mid) = edge
            edge_key = tuple(sorted([v1, v2]))
            if edge_key not in edge_to_elements:
                edge_to_elements[edge_key] = []
            edge_to_elements[edge_key].append((elem_idx, mid))
    for (edge_key, elem_list) in edge_to_elements.items():
        if len(elem_list) == 2:
            mid1 = elem_list[0][1]
            mid2 = elem_list[1][1]
            assert mid1 == mid2

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