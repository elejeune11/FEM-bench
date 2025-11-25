def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    num_nodes = npx * npy
    num_elements = 2 * nx * ny
    assert coords.shape[0] == num_nodes, 'Incorrect number of nodes'
    assert connect.shape[0] == num_elements, 'Incorrect number of elements'
    assert coords.shape == (num_nodes, 2), 'Incorrect coords shape'
    assert connect.shape == (num_elements, 6), 'Incorrect connect shape'
    assert coords.dtype == np.float64, 'Incorrect coords dtype'
    assert connect.dtype == np.int64, 'Incorrect connect dtype'
    bottom_left_id = 0 * npx + 0
    bottom_right_id = 0 * npx + (npx - 1)
    top_left_id = (npy - 1) * npx + 0
    top_right_id = (npy - 1) * npx + (npx - 1)
    assert_allclose(coords[bottom_left_id], [xl, yl])
    assert_allclose(coords[bottom_right_id], [xh, yl])
    assert_allclose(coords[top_left_id], [xl, yh])
    assert_allclose(coords[top_right_id], [xh, yh])
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x_steps = xl + 0.5 * dx * np.arange(npx)
    y_steps = yl + 0.5 * dy * np.arange(npy)
    (xx, yy) = np.meshgrid(x_steps, y_steps, indexing='xy')
    expected_coords = np.vstack([xx.ravel(), yy.ravel()]).T
    assert_allclose(coords, expected_coords, err_msg='Node coordinates do not form the expected lattice')
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert_array_equal(coords, coords2, err_msg='Coords are not deterministic')
    assert_array_equal(connect, connect2, err_msg='Connectivity is not deterministic')

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 0.0, 3.0, 1.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    num_nodes = coords.shape[0]
    assert np.all(connect >= 0) and np.all(connect < num_nodes), 'Connectivity indices are out of bounds'
    assert all((len(set(row)) == 6 for row in connect)), 'Node indices are not unique within an element'
    corners = connect[:, :3]
    p1 = coords[corners[:, 0]]
    p2 = coords[corners[:, 1]]
    p3 = coords[corners[:, 2]]
    signed_areas = (p2[:, 0] - p1[:, 0]) * (p3[:, 1] - p1[:, 1]) - (p3[:, 0] - p1[:, 0]) * (p2[:, 1] - p1[:, 1])
    assert np.all(signed_areas > 1e-12), 'Element corner nodes are not consistently counter-clockwise'
    (N1, N2, N3, N4, N5, N6) = connect.T
    expected_N4_coords = (coords[N1] + coords[N2]) / 2.0
    expected_N5_coords = (coords[N2] + coords[N3]) / 2.0
    expected_N6_coords = (coords[N3] + coords[N1]) / 2.0
    assert_allclose(coords[N4], expected_N4_coords, err_msg='Midside node N4 is not at the midpoint of (N1, N2)')
    assert_allclose(coords[N5], expected_N5_coords, err_msg='Midside node N5 is not at the midpoint of (N2, N3)')
    assert_allclose(coords[N6], expected_N6_coords, err_msg='Midside node N6 is not at the midpoint of (N3, N1)')

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