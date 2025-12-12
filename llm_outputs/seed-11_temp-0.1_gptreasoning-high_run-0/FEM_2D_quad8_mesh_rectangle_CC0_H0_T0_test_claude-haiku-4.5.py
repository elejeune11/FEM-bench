def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.allclose(coords1, coords2), 'Repeated calls should yield identical coordinates'
    assert np.array_equal(connect1, connect2), 'Repeated calls should yield identical connectivity'
    expected_nnodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_nelems = nx * ny
    assert coords1.shape == (expected_nnodes, 2), f'Expected coords shape {(expected_nnodes, 2)}, got {coords1.shape}'
    assert connect1.shape == (expected_nelems, 8), f'Expected connect shape {(expected_nelems, 8)}, got {connect1.shape}'
    assert coords1.dtype == np.float64, f'Expected coords dtype float64, got {coords1.dtype}'
    assert connect1.dtype == np.int64, f'Expected connect dtype int64, got {connect1.dtype}'
    assert coords1.shape[0] == expected_nnodes, f'Expected {expected_nnodes} nodes, got {coords1.shape[0]}'
    assert connect1.shape[0] == expected_nelems, f'Expected {expected_nelems} elements, got {connect1.shape[0]}'
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    corner_coords = [(xl, yl), (xh, yl), (xh, yh), (xl, yh)]
    for (corner_x, corner_y) in corner_coords:
        found = False
        for coord in coords1:
            if np.isclose(coord[0], corner_x) and np.isclose(coord[1], corner_y):
                found = True
                break
        assert found, f'Corner ({corner_x}, {corner_y}) not found in coordinates'
    half_dx = dx / 2.0
    half_dy = dy / 2.0
    for coord in coords1:
        x_frac = (coord[0] - xl) / half_dx
        y_frac = (coord[1] - yl) / half_dy
        assert np.isclose(x_frac, round(x_frac)), f'X coordinate {coord[0]} not on half-step grid'
        assert np.isclose(y_frac, round(y_frac)), f'Y coordinate {coord[1]} not on half-step grid'

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    nelems = connect.shape[0]
    assert np.all(connect >= 0), 'Connectivity indices must be non-negative'
    assert np.all(connect < nnodes), f'Connectivity indices must be < {nnodes}'
    for (elem_idx, elem) in enumerate(connect):
        (n1, n2, n3, n4) = (elem[0], elem[1], elem[2], elem[3])
        p1 = coords[n1]
        p2 = coords[n2]
        p3 = coords[n3]
        p4 = coords[n4]
        area = 0.5 * ((p2[0] - p1[0]) * (p4[1] - p1[1]) - (p4[0] - p1[0]) * (p2[1] - p1[1]) + (p3[0] - p2[0]) * (p4[1] - p2[1]) - (p4[0] - p2[0]) * (p3[1] - p2[1]))
        assert area > 0, f'Element {elem_idx} has non-positive area (not counter-clockwise)'
    for (elem_idx, elem) in enumerate(connect):
        (n1, n2, n3, n4) = (elem[0], elem[1], elem[2], elem[3])
        (n5, n6, n7, n8) = (elem[4], elem[5], elem[6], elem[7])
        (p1, p2, p3, p4) = (coords[n1], coords[n2], coords[n3], coords[n4])
        (p5, p6, p7, p8) = (coords[n5], coords[n6], coords[n7], coords[n8])
        expected_p5 = (p1 + p2) / 2.0
        assert np.allclose(p5, expected_p5), f'Element {elem_idx}: N5 not midpoint of N1-N2'
        expected_p6 = (p2 + p3) / 2.0
        assert np.allclose(p6, expected_p6), f'Element {elem_idx}: N6 not midpoint of N2-N3'
        expected_p7 = (p3 + p4) / 2.0
        assert np.allclose(p7, expected_p7), f'Element {elem_idx}: N7 not midpoint of N3-N4'
        expected_p8 = (p4 + p1) / 2.0
        assert np.allclose(p8, expected_p8), f'Element {elem_idx}: N8 not midpoint of N4-N1'
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    for cy in range(ny):
        for cx in range(nx):
            elem_idx = cy * nx + cx
            elem = connect[elem_idx]
            (n1, n2, n3, n4) = (elem[0], elem[1], elem[2], elem[3])
            if cx < nx - 1:
                right_elem_idx = cy * nx + (cx + 1)
                right_elem = connect[right_elem_idx]
                (right_n1, right_n4) = (right_elem[0], right_elem[3])
                assert elem[1] == right_n1, f'Right neighbor conformity failed at ({cx},{cy}): N2 != right_N1'
                assert elem[2] == right_n4, f'Right neighbor conformity failed at ({cx},{cy}): N3 != right_N4'
            if cy < ny - 1:
                top_elem_idx = (cy + 1) * nx + cx
                top_elem = connect[top_elem_idx]
                (top_n1, top_n2) = (top_elem[0], top_elem[1])
                assert elem[3] == top_n1, f'Top neighbor conformity failed at ({cx},{cy}): N4 != top_N1'
                assert elem[2] == top_n2, f'Top neighbor conformity failed at ({cx},{cy}): N3 != top_N2'

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
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 1, 1)