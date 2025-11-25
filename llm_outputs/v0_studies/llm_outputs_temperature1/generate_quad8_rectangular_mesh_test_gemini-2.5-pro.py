def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    expected_nnodes = npx * npy - nx * ny
    expected_nelems = nx * ny
    assert coords.shape[0] == expected_nnodes, 'Incorrect number of nodes'
    assert connect.shape[0] == expected_nelems, 'Incorrect number of elements'
    assert coords.shape == (expected_nnodes, 2), 'Incorrect coords shape'
    assert connect.shape == (expected_nelems, 8), 'Incorrect connect shape'
    assert coords.dtype == np.float64, 'Incorrect coords dtype'
    assert connect.dtype == np.int64, 'Incorrect connect dtype'
    domain_corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]])
    for corner in domain_corners:
        assert np.any(np.all(np.isclose(coords, corner), axis=1)), 'Domain corner not found'
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x_grid = xl + 0.5 * dx * np.arange(npx)
    y_grid = yl + 0.5 * dy * np.arange(npy)
    (xx, yy) = np.meshgrid(x_grid, y_grid, indexing='xy')
    all_refined_coords = np.vstack([xx.ravel(), yy.ravel()]).T
    mask = np.ones((npy, npx), dtype=bool)
    mask[1::2, 1::2] = False
    expected_coords = all_refined_coords[mask.ravel()]
    assert np.allclose(coords, expected_coords), 'Node coordinates do not match specification'
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2), 'Coordinates are not deterministic'
    assert np.array_equal(connect, connect2), 'Connectivity is not deterministic'

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 0.0, 3.0, 1.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    assert np.all(connect >= 0) and np.all(connect < nnodes), 'Connectivity indices out of range'
    assert all((len(set(row)) == 8 for row in connect)), 'Node indices are not unique within an element'
    corner_indices = connect[:, :4]
    corner_coords = coords[corner_indices]
    x = corner_coords[:, :, 0]
    y = corner_coords[:, :, 1]
    area = 0.5 * (x[:, 0] * y[:, 1] - y[:, 0] * x[:, 1] + x[:, 1] * y[:, 2] - y[:, 1] * x[:, 2] + x[:, 2] * y[:, 3] - y[:, 2] * x[:, 3] + x[:, 3] * y[:, 0] - y[:, 3] * x[:, 0])
    assert np.all(area > 1e-09), 'Element corner nodes are not consistently counter-clockwise'
    (c1, c2, c3, c4) = (coords[connect[:, i]] for i in range(4))
    (c5, c6, c7, c8) = (coords[connect[:, i]] for i in range(4, 8))
    assert np.allclose(c5, 0.5 * (c1 + c2)), 'Midside node N5 is incorrectly placed'
    assert np.allclose(c6, 0.5 * (c2 + c3)), 'Midside node N6 is incorrectly placed'
    assert np.allclose(c7, 0.5 * (c3 + c4)), 'Midside node N7 is incorrectly placed'
    assert np.allclose(c8, 0.5 * (c4 + c1)), 'Midside node N8 is incorrectly placed'
    for cy in range(ny):
        for cx in range(nx):
            elem_idx = cy * nx + cx
            current_elem = connect[elem_idx]
            if cx < nx - 1:
                right_elem_idx = cy * nx + (cx + 1)
                right_elem = connect[right_elem_idx]
                assert current_elem[1] == right_elem[0], 'Horizontal conformity failed (N2/N1)'
                assert current_elem[2] == right_elem[3], 'Horizontal conformity failed (N3/N4)'
                assert current_elem[5] == right_elem[7], 'Horizontal conformity failed (N6/N8)'
            if cy < ny - 1:
                top_elem_idx = (cy + 1) * nx + cx
                top_elem = connect[top_elem_idx]
                assert current_elem[3] == top_elem[0], 'Vertical conformity failed (N4/N1)'
                assert current_elem[2] == top_elem[1], 'Vertical conformity failed (N3/N2)'
                assert current_elem[6] == top_elem[4], 'Vertical conformity failed (N7/N5)'

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 0, ny)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, -1, ny)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, nx, 0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, nx, -5)
    with pytest.raises(ValueError):
        fcn(1.0, yl, 1.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(1.1, yl, 1.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 1.0, xh, 1.0, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 1.1, xh, 1.0, nx, ny)