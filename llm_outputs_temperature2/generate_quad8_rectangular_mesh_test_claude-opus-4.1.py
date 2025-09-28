def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_elements = nx * ny
    assert coords.shape[0] == expected_nodes
    assert connect.shape[0] == expected_elements
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elements, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    corners = [(xl, yl), (xh, yl), (xh, yh), (xl, yh)]
    for corner in corners:
        assert np.any(np.all(np.isclose(coords, corner), axis=1))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy
    x_refined = xl + np.arange(2 * nx + 1) * half_dx
    y_refined = yl + np.arange(2 * ny + 1) * half_dy
    (xx, yy) = np.meshgrid(x_refined, y_refined, indexing='xy')
    all_coords = np.column_stack([xx.ravel(), yy.ravel()])
    mask = np.ones(len(all_coords), dtype=bool)
    for iy in range(2 * ny + 1):
        for ix in range(2 * nx + 1):
            if ix % 2 == 1 and iy % 2 == 1:
                mask[iy * (2 * nx + 1) + ix] = False
    expected_coords = all_coords[mask]
    assert np.allclose(coords, expected_coords)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    num_nodes = coords.shape[0]
    num_elements = connect.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < num_nodes)
    for elem in connect:
        assert len(np.unique(elem)) == 8
    for elem in connect:
        corners = elem[:4]
        corner_coords = coords[corners]
        area = 0.0
        for i in range(4):
            j = (i + 1) % 4
            area += corner_coords[i, 0] * corner_coords[j, 1]
            area -= corner_coords[j, 0] * corner_coords[i, 1]
        area *= 0.5
        assert area > 0
    for elem in connect:
        (n1, n2, n3, n4, n5, n6, n7, n8) = elem
        expected_n5 = 0.5 * (coords[n1] + coords[n2])
        assert np.allclose(coords[n5], expected_n5)
        expected_n6 = 0.5 * (coords[n2] + coords[n3])
        assert np.allclose(coords[n6], expected_n6)
        expected_n7 = 0.5 * (coords[n3] + coords[n4])
        assert np.allclose(coords[n7], expected_n7)
        expected_n8 = 0.5 * (coords[n4] + coords[n1])
        assert np.allclose(coords[n8], expected_n8)
    edge_map = {}
    for (elem_idx, elem) in enumerate(connect):
        corners = elem[:4]
        edges = [(corners[0], corners[1]), (corners[1], corners[2]), (corners[2], corners[3]), (corners[3], corners[0])]
        for edge in edges:
            sorted_edge = tuple(sorted(edge))
            if sorted_edge not in edge_map:
                edge_map[sorted_edge] = []
            edge_map[sorted_edge].append(elem_idx)
    for (edge, elems) in edge_map.items():
        assert len(elems) <= 2

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