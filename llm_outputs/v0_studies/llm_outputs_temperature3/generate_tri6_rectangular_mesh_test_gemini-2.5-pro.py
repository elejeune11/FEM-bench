def test_tri6_mesh_basic_structure_and_determinism(fcn):
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
    assert coords.shape == (expected_num_nodes, 2)
    assert connect.shape == (expected_num_elements, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    bottom_left_id = 0
    bottom_right_id = npx - 1
    top_left_id = (npy - 1) * npx
    top_right_id = (npy - 1) * npx + (npx - 1)
    np.testing.assert_allclose(coords[bottom_left_id], [xl, yl])
    np.testing.assert_allclose(coords[bottom_right_id], [xh, yl])
    np.testing.assert_allclose(coords[top_left_id], [xl, yh])
    np.testing.assert_allclose(coords[top_right_id], [xh, yh])
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x_steps = xl + 0.5 * dx * np.arange(npx)
    y_steps = yl + 0.5 * dy * np.arange(npy)
    (xx, yy) = np.meshgrid(x_steps, y_steps, indexing='xy')
    expected_coords = np.vstack([xx.ravel(), yy.ravel()]).T
    np.testing.assert_allclose(coords, expected_coords)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    np.testing.assert_array_equal(coords, coords2)
    np.testing.assert_array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (1.0, 2.0, 5.0, 3.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    num_nodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < num_nodes)
    for row in connect:
        assert len(set(row)) == 6
    for element in connect:
        (n1, n2, n3, n4, n5, n6) = element
        (p1, p2, p3) = (coords[n1], coords[n2], coords[n3])
        signed_area_x2 = p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])
        assert signed_area_x2 > 1e-12
        np.testing.assert_allclose(coords[n4], (p1 + p2) / 2.0)
        np.testing.assert_allclose(coords[n5], (p2 + p3) / 2.0)
        np.testing.assert_allclose(coords[n6], (p3 + p1) / 2.0)

def test_tri6_mesh_invalid_inputs(fcn):
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