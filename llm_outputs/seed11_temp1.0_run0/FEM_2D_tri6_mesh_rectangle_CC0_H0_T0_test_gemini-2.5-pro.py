def test_tri6_mesh_basic_structure_and_determinism(fcn: Callable):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    expected_num_nodes = npx * npy
    expected_num_elements = 2 * nx * ny
    assert coords.shape == (expected_num_nodes, 2), 'Incorrect coords shape'
    assert connect.shape == (expected_num_elements, 6), 'Incorrect connect shape'
    assert coords.dtype == np.float64, 'Incorrect coords dtype'
    assert connect.dtype == np.int64, 'Incorrect connect dtype'
    np.testing.assert_allclose(coords[0], [xl, yl])
    np.testing.assert_allclose(coords[npx - 1], [xh, yl])
    np.testing.assert_allclose(coords[(npy - 1) * npx], [xl, yh])
    np.testing.assert_allclose(coords[-1], [xh, yh])
    x_steps = np.linspace(xl, xh, npx)
    y_steps = np.linspace(yl, yh, npy)
    (xx, yy) = np.meshgrid(x_steps, y_steps, indexing='xy')
    expected_coords = np.vstack([xx.ravel(), yy.ravel()]).T
    np.testing.assert_allclose(coords, expected_coords)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    np.testing.assert_array_equal(coords, coords2)
    np.testing.assert_array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn: Callable):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (1.0, 2.0, 5.0, 8.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    num_nodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < num_nodes)
    for i in range(connect.shape[0]):
        assert len(np.unique(connect[i, :])) == 6, f'Element {i} has duplicate node IDs'
    p1_coords = coords[connect[:, 0]]
    p2_coords = coords[connect[:, 1]]
    p3_coords = coords[connect[:, 2]]
    signed_area_x2 = (p2_coords[:, 0] - p1_coords[:, 0]) * (p3_coords[:, 1] - p1_coords[:, 1]) - (p2_coords[:, 1] - p1_coords[:, 1]) * (p3_coords[:, 0] - p1_coords[:, 0])
    assert np.all(signed_area_x2 > 1e-12), 'Found a clockwise or degenerate corner node ordering'
    expected_n4_coords = 0.5 * (coords[connect[:, 0]] + coords[connect[:, 1]])
    np.testing.assert_allclose(coords[connect[:, 3]], expected_n4_coords)
    expected_n5_coords = 0.5 * (coords[connect[:, 1]] + coords[connect[:, 2]])
    np.testing.assert_allclose(coords[connect[:, 4]], expected_n5_coords)
    expected_n6_coords = 0.5 * (coords[connect[:, 2]] + coords[connect[:, 0]])
    np.testing.assert_allclose(coords[connect[:, 5]], expected_n6_coords)

def test_tri6_mesh_invalid_inputs(fcn: Callable):
    """Validate error handling for invalid inputs.
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