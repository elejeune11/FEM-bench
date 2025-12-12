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
    expected_nelements = nx * ny
    assert coords.shape == (expected_nnodes, 2), f'Expected coords shape {(expected_nnodes, 2)}, got {coords.shape}'
    assert connect.shape == (expected_nelements, 8), f'Expected connect shape {(expected_nelements, 8)}, got {connect.shape}'
    assert coords.dtype == np.float64, f'Expected coords dtype float64, got {coords.dtype}'
    assert connect.dtype == np.int64, f'Expected connect dtype int64, got {connect.dtype}'
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    domain_corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]])
    for corner in domain_corners:
        distances = np.linalg.norm(coords - corner, axis=1)
        assert np.min(distances) < 1e-10, f'Corner {corner} not found in coords'
    half_dx = dx / 2
    half_dy = dy / 2
    for i in range(coords.shape[0]):
        (x, y) = coords[i]
        x_steps = (x - xl) / half_dx
        y_steps = (y - yl) / half_dy
        assert abs(x_steps - round(x_steps)) < 1e-10, f'x coordinate {x} not on half-step lattice'
        assert abs(y_steps - round(y_steps)) < 1e-10, f'y coordinate {y} not on half-step lattice'
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2), 'Coords not deterministic'
    assert np.array_equal(connect, connect2), 'Connect not deterministic'

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    nelements = connect.shape[0]
    assert np.all(connect >= 0), 'Negative node indices in connectivity'
    assert np.all(connect < nnodes), 'Node indices out of range in connectivity'
    for elem_idx in range(nelements):
        elem_nodes = connect[elem_idx]
        assert len(np.unique(elem_nodes)) == 8, f'Element {elem_idx} has duplicate node indices'
    for elem_idx in range(nelements):
        (n1, n2, n3, n4) = connect[elem_idx, :4]
        p1 = coords[n1]
        p2 = coords[n2]
        p3 = coords[n3]
        p4 = coords[n4]
        area = 0.5 * (p1[0] * p2[1] - p2[0] * p1[1] + (p2[0] * p3[1] - p3[0] * p2[1]) + (p3[0] * p4[1] - p4[0] * p3[1]) + (p4[0] * p1[1] - p1[0] * p4[1]))
        assert area > 0, f'Element {elem_idx} has non-positive area (not CCW): {area}'
    for elem_idx in range(nelements):
        (n1, n2, n3, n4, n5, n6, n7, n8) = connect[elem_idx]
        expected_n5 = (coords[n1] + coords[n2]) / 2
        assert np.allclose(coords[n5], expected_n5), f'Element {elem_idx}: N5 not midpoint of N1-N2'
        expected_n6 = (coords[n2] + coords[n3]) / 2
        assert np.allclose(coords[n6], expected_n6), f'Element {elem_idx}: N6 not midpoint of N2-N3'
        expected_n7 = (coords[n3] + coords[n4]) / 2
        assert np.allclose(coords[n7], expected_n7), f'Element {elem_idx}: N7 not midpoint of N3-N4'
        expected_n8 = (coords[n4] + coords[n1]) / 2
        assert np.allclose(coords[n8], expected_n8), f'Element {elem_idx}: N8 not midpoint of N4-N1'
    elem_map = {}
    for elem_idx in range(nelements):
        cy = elem_idx // nx
        cx = elem_idx % nx
        elem_map[cx, cy] = elem_idx
    for cy in range(ny):
        for cx in range(nx - 1):
            left_elem = elem_map[cx, cy]
            right_elem = elem_map[cx + 1, cy]
            left_n2 = connect[left_elem, 1]
            left_n3 = connect[left_elem, 2]
            left_n6 = connect[left_elem, 5]
            right_n1 = connect[right_elem, 0]
            right_n4 = connect[right_elem, 3]
            right_n8 = connect[right_elem, 7]
            assert left_n2 == right_n1, f'Non-conforming: left N2 != right N1'
            assert left_n3 == right_n4, f'Non-conforming: left N3 != right N4'
            assert left_n6 == right_n8, f'Non-conforming: left N6 != right N8'
    for cy in range(ny - 1):
        for cx in range(nx):
            bottom_elem = elem_map[cx, cy]
            top_elem = elem_map[cx, cy + 1]
            bottom_n3 = connect[bottom_elem, 2]
            bottom_n4 = connect[bottom_elem, 3]
            bottom_n7 = connect[bottom_elem, 6]
            top_n1 = connect[top_elem, 0]
            top_n2 = connect[top_elem, 1]
            top_n5 = connect[top_elem, 4]
            assert bottom_n4 == top_n1, f'Non-conforming: bottom N4 != top N1'
            assert bottom_n3 == top_n2, f'Non-conforming: bottom N3 != top N2'
            assert bottom_n7 == top_n5, f'Non-conforming: bottom N7 != top N5'

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