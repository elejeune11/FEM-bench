def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nnodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_nelems = nx * ny
    assert coords.shape[0] == expected_nnodes
    assert connect.shape[0] == expected_nelems
    assert coords.shape == (expected_nnodes, 2)
    assert connect.shape == (expected_nelems, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    assert np.any(np.isclose(x_coords, xl))
    assert np.any(np.isclose(x_coords, xh))
    assert np.any(np.isclose(y_coords, yl))
    assert np.any(np.isclose(y_coords, yh))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    expected_x = np.array([xl + 0.5 * dx * i for i in range(2 * nx + 1)])
    expected_y = np.array([yl + 0.5 * dy * i for i in range(2 * ny + 1)])
    unique_x = np.unique(x_coords)
    unique_y = np.unique(y_coords)
    expected_x_filtered = []
    expected_y_filtered = []
    for iy in range(2 * ny + 1):
        for ix in range(2 * nx + 1):
            if not (ix % 2 == 1 and iy % 2 == 1):
                if expected_x[ix] not in expected_x_filtered:
                    expected_x_filtered.append(expected_x[ix])
                if expected_y[iy] not in expected_y_filtered:
                    expected_y_filtered.append(expected_y[iy])
    expected_x_filtered = sorted(list(set(expected_x_filtered)))
    expected_y_filtered = sorted(list(set(expected_y_filtered)))
    assert np.allclose(sorted(unique_x), expected_x_filtered)
    assert np.allclose(sorted(unique_y), expected_y_filtered)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (1.0, 2.0, 4.0, 5.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    max_node_id = coords.shape[0] - 1
    assert np.all(connect >= 0)
    assert np.all(connect <= max_node_id)
    for elem in connect:
        assert len(np.unique(elem)) == 8
    for elem in connect:
        (n1, n2, n3, n4) = elem[:4]
        (p1, p2, p3, p4) = (coords[n1], coords[n2], coords[n3], coords[n4])
        area = 0.5 * (p1[0] * (p2[1] - p4[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p4[1] - p2[1]) + p4[0] * (p1[1] - p3[1]))
        assert area > 0
    for elem in connect:
        (n1, n2, n3, n4, n5, n6, n7, n8) = elem
        (p1, p2, p3, p4, p5, p6, p7, p8) = coords[elem]
        expected_p5 = 0.5 * (p1 + p2)
        assert np.allclose(p5, expected_p5)
        expected_p6 = 0.5 * (p2 + p3)
        assert np.allclose(p6, expected_p6)
        expected_p7 = 0.5 * (p3 + p4)
        assert np.allclose(p7, expected_p7)
        expected_p8 = 0.5 * (p4 + p1)
        assert np.allclose(p8, expected_p8)
    edge_to_elems = {}
    for (elem_idx, elem) in enumerate(connect):
        (n1, n2, n3, n4, n5, n6, n7, n8) = elem
        edges = [(min(n1, n2), max(n1, n2), n5), (min(n2, n3), max(n2, n3), n6), (min(n3, n4), max(n3, n4), n7), (min(n4, n1), max(n4, n1), n8)]
        for edge in edges:
            corner_pair = (edge[0], edge[1])
            midside = edge[2]
            if corner_pair not in edge_to_elems:
                edge_to_elems[corner_pair] = []
            edge_to_elems[corner_pair].append((elem_idx, midside))
    for (corner_pair, elem_data) in edge_to_elems.items():
        if len(elem_data) > 1:
            midside_ids = [data[1] for data in elem_data]
            assert len(set(midside_ids)) == 1

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
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