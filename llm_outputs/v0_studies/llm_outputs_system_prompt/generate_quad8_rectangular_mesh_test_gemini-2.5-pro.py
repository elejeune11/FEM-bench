def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_num_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_num_elements = nx * ny
    assert coords.shape[0] == expected_num_nodes, 'Incorrect number of nodes'
    assert connect.shape[0] == expected_num_elements, 'Incorrect number of elements'
    assert coords.shape == (expected_num_nodes, 2), 'Incorrect coords shape'
    assert connect.shape == (expected_num_elements, 8), 'Incorrect connect shape'
    assert coords.dtype == np.float64, 'Incorrect coords dtype'
    assert connect.dtype == np.int64, 'Incorrect connect dtype'
    domain_corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]])
    for corner in domain_corners:
        assert np.any(np.all(np.isclose(coords, corner), axis=1)), f'Domain corner {corner} not found'
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy
    x_coords = coords[:, 0]
    y_coords = coords[:, 1]
    x_indices = np.round((x_coords - xl) / half_dx)
    y_indices = np.round((y_coords - yl) / half_dy)
    assert np.allclose(x_coords, xl + x_indices * half_dx)
    assert np.allclose(y_coords, yl + y_indices * half_dy)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2), 'Coordinates are not deterministic'
    assert np.array_equal(connect, connect2), 'Connectivity is not deterministic'

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (1.0, 2.0, 5.0, 3.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    num_nodes = coords.shape[0]
    assert np.all(connect >= 0) and np.all(connect < num_nodes), 'Connectivity indices out of range'
    assert all((len(set(row)) == 8 for row in connect)), 'Node IDs within an element are not unique'
    for el_nodes in connect:
        c_nodes = coords[el_nodes[:4]]
        area = 0.5 * (c_nodes[0, 0] * c_nodes[1, 1] - c_nodes[1, 0] * c_nodes[0, 1] + c_nodes[1, 0] * c_nodes[2, 1] - c_nodes[2, 0] * c_nodes[1, 1] + c_nodes[2, 0] * c_nodes[3, 1] - c_nodes[3, 0] * c_nodes[2, 1] + c_nodes[3, 0] * c_nodes[0, 1] - c_nodes[0, 0] * c_nodes[3, 1])
        assert area > 1e-09, 'Element corner nodes are not in CCW order or have zero area'
    for el_nodes in connect:
        (n1, n2, n3, n4, n5, n6, n7, n8) = el_nodes
        assert np.allclose(coords[n5], (coords[n1] + coords[n2]) / 2.0), 'Midside N5 position is incorrect'
        assert np.allclose(coords[n6], (coords[n2] + coords[n3]) / 2.0), 'Midside N6 position is incorrect'
        assert np.allclose(coords[n7], (coords[n3] + coords[n4]) / 2.0), 'Midside N7 position is incorrect'
        assert np.allclose(coords[n8], (coords[n4] + coords[n1]) / 2.0), 'Midside N8 position is incorrect'
    for cy in range(ny):
        for cx in range(nx):
            el_idx = cy * nx + cx
            if cx < nx - 1:
                neighbor_idx = cy * nx + (cx + 1)
                assert connect[el_idx, 1] == connect[neighbor_idx, 0], 'Right edge corner node mismatch (N2/N1)'
                assert connect[el_idx, 2] == connect[neighbor_idx, 3], 'Right edge corner node mismatch (N3/N4)'
                assert connect[el_idx, 5] == connect[neighbor_idx, 7], 'Right edge midside node mismatch (N6/N8)'
            if cy < ny - 1:
                neighbor_idx = (cy + 1) * nx + cx
                assert connect[el_idx, 3] == connect[neighbor_idx, 0], 'Top edge corner node mismatch (N4/N1)'
                assert connect[el_idx, 2] == connect[neighbor_idx, 1], 'Top edge corner node mismatch (N3/N2)'
                assert connect[el_idx, 6] == connect[neighbor_idx, 4], 'Top edge midside node mismatch (N7/N5)'

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (1, 1)
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
        fcn(1.1, yl, 1.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 1.0, xh, 1.0, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 1.1, xh, 1.0, nx, ny)