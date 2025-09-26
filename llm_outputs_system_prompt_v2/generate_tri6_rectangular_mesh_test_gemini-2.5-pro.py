def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    num_nodes = npx * npy
    num_elements = 2 * nx * ny
    assert coords.shape == (num_nodes, 2), 'Incorrect coords shape'
    assert connect.shape == (num_elements, 6), 'Incorrect connect shape'
    assert coords.dtype == np.float64, 'Incorrect coords dtype'
    assert connect.dtype == np.int64, 'Incorrect connect dtype'
    assert np.allclose(coords[0], [xl, yl])
    assert np.allclose(coords[2 * nx], [xh, yl])
    assert np.allclose(coords[2 * ny * npx], [xl, yh])
    assert np.allclose(coords[2 * ny * npx + 2 * nx], [xh, yh])
    x_steps = np.linspace(xl, xh, npx)
    y_steps = np.linspace(yl, yh, npy)
    (xx, yy) = np.meshgrid(x_steps, y_steps, indexing='xy')
    expected_coords = np.vstack([xx.ravel(), yy.ravel()]).T
    assert np.allclose(coords, expected_coords)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (1.0, 2.0, 5.0, 8.0)
    (nx, ny) = (3, 4)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    num_nodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < num_nodes)
    for i in range(connect.shape[0]):
        assert len(np.unique(connect[i, :])) == 6
    corner_nodes = connect[:, :3]
    p1_corners = coords[corner_nodes[:, 0]]
    p2_corners = coords[corner_nodes[:, 1]]
    p3_corners = coords[corner_nodes[:, 2]]
    areas = 0.5 * (p1_corners[:, 0] * (p2_corners[:, 1] - p3_corners[:, 1]) + p2_corners[:, 0] * (p3_corners[:, 1] - p1_corners[:, 1]) + p3_corners[:, 0] * (p1_corners[:, 1] - p2_corners[:, 1]))
    assert np.all(areas > 1e-12)
    p1 = coords[connect[:, 0]]
    p2 = coords[connect[:, 1]]
    p3 = coords[connect[:, 2]]
    p4 = coords[connect[:, 3]]
    p5 = coords[connect[:, 4]]
    p6 = coords[connect[:, 5]]
    assert np.allclose(p4, 0.5 * (p1 + p2))
    assert np.allclose(p5, 0.5 * (p2 + p3))
    assert np.allclose(p6, 0.5 * (p3 + p1))
    unique_coords = np.unique(coords, axis=0)
    assert unique_coords.shape == coords.shape

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