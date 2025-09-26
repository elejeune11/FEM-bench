def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    expected_num_nodes = npx * npy
    expected_num_elements = 2 * nx * ny
    assert coords.shape[0] == expected_num_nodes, 'Incorrect number of nodes'
    assert connect.shape[0] == expected_num_elements, 'Incorrect number of elements'
    assert coords.shape == (expected_num_nodes, 2), 'Incorrect coords shape'
    assert connect.shape == (expected_num_elements, 6), 'Incorrect connect shape'
    assert coords.dtype == np.float64, 'Incorrect coords dtype'
    assert connect.dtype == np.int64, 'Incorrect connect dtype'
    assert np.min(coords[:, 0]) == xl
    assert np.max(coords[:, 0]) == xh
    assert np.min(coords[:, 1]) == yl
    assert np.max(coords[:, 1]) == yh
    x_coords = np.linspace(xl, xh, npx)
    y_coords = np.linspace(yl, yh, npy)
    (xx, yy) = np.meshgrid(x_coords, y_coords, indexing='xy')
    expected_coords = np.vstack([xx.ravel(), yy.ravel()]).T
    np.testing.assert_allclose(coords, expected_coords, atol=1e-15, rtol=1e-15)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    np.testing.assert_array_equal(coords, coords2)
    np.testing.assert_array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (1.0, 2.0, 5.0, 8.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    num_nodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < num_nodes)
    for row in connect:
        assert len(np.unique(row)) == 6, 'Node indices in an element must be unique'
    c1 = coords[connect[:, 0]]
    c2 = coords[connect[:, 1]]
    c3 = coords[connect[:, 2]]
    signed_area = 0.5 * (c1[:, 0] * (c2[:, 1] - c3[:, 1]) + c2[:, 0] * (c3[:, 1] - c1[:, 1]) + c3[:, 0] * (c1[:, 1] - c2[:, 1]))
    assert np.all(signed_area > 1e-12), 'Corner nodes are not all CCW'
    c4 = coords[connect[:, 3]]
    c5 = coords[connect[:, 4]]
    c6 = coords[connect[:, 5]]
    np.testing.assert_allclose(c4, 0.5 * (c1 + c2), err_msg='Midside node N4 is incorrectly placed')
    np.testing.assert_allclose(c5, 0.5 * (c2 + c3), err_msg='Midside node N5 is incorrectly placed')
    np.testing.assert_allclose(c6, 0.5 * (c3 + c1), err_msg='Midside node N6 is incorrectly placed')
    edge_counts = {}
    for el in connect:
        nodes = [el[0], el[3], el[1], el[4], el[2], el[5]]
        for i in range(6):
            (u, v) = (nodes[i], nodes[(i + 1) % 6])
            edge = tuple(sorted((u, v)))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    for (edge, count) in edge_counts.items():
        (u, v) = edge
        (c_u, c_v) = (coords[u], coords[v])
        on_xl = np.isclose(c_u[0], xl) and np.isclose(c_v[0], xl)
        on_xh = np.isclose(c_u[0], xh) and np.isclose(c_v[0], xh)
        on_yl = np.isclose(c_u[1], yl) and np.isclose(c_v[1], yl)
        on_yh = np.isclose(c_u[1], yh) and np.isclose(c_v[1], yh)
        is_boundary = on_xl or on_xh or on_yl or on_yh
        if is_boundary:
            assert count == 1, f'Boundary edge {edge} shared by {count} elements'
        else:
            assert count == 2, f'Internal edge {edge} shared by {count} elements'

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
        fcn(1.1, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.1, 1.0, 1.0, 1, 1)