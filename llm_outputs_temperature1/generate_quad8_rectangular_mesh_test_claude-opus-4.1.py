def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nnodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    assert coords.shape[0] == expected_nnodes
    assert coords.shape == (expected_nnodes, 2)
    assert connect.shape == (nx * ny, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    corners = []
    for x in [xl, xh]:
        for y in [yl, yh]:
            corners.append([x, y])
    corners = np.array(corners)
    for corner in corners:
        assert np.any(np.all(np.isclose(coords, corner), axis=1))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy
    for coord in coords:
        x_steps = (coord[0] - xl) / half_dx
        y_steps = (coord[1] - yl) / half_dy
        assert np.isclose(x_steps, np.round(x_steps))
        assert np.isclose(y_steps, np.round(y_steps))
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
    assert np.all(connect >= 0)
    assert np.all(connect < coords.shape[0])
    for elem in connect:
        assert len(np.unique(elem)) == 8
    for elem in connect:
        (n1, n2, n3, n4) = (elem[0], elem[1], elem[2], elem[3])
        (p1, p2, p3, p4) = (coords[n1], coords[n2], coords[n3], coords[n4])
        area = 0.5 * ((p2[0] - p1[0]) * (p4[1] - p1[1]) - (p4[0] - p1[0]) * (p2[1] - p1[1]))
        area += 0.5 * ((p3[0] - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (p3[1] - p2[1]))
        area += 0.5 * ((p4[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p4[1] - p3[1]))
        area += 0.5 * ((p1[0] - p4[0]) * (p3[1] - p4[1]) - (p3[0] - p4[0]) * (p1[1] - p4[1]))
        assert area > 0
    for elem in connect:
        (n1, n2, n3, n4, n5, n6, n7, n8) = elem
        assert np.allclose(coords[n5], 0.5 * (coords[n1] + coords[n2]))
        assert np.allclose(coords[n6], 0.5 * (coords[n2] + coords[n3]))
        assert np.allclose(coords[n7], 0.5 * (coords[n3] + coords[n4]))
        assert np.allclose(coords[n8], 0.5 * (coords[n4] + coords[n1]))
    for cy in range(ny):
        for cx in range(nx - 1):
            elem_idx = cy * nx + cx
            next_elem_idx = cy * nx + cx + 1
            right_nodes = [connect[elem_idx][1], connect[elem_idx][5], connect[elem_idx][2]]
            left_nodes = [connect[next_elem_idx][0], connect[next_elem_idx][7], connect[next_elem_idx][3]]
            assert right_nodes[0] == left_nodes[0]
            assert right_nodes[2] == left_nodes[2]

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