def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords1.shape == (coords1.shape[0], 2)
    assert coords1.dtype == np.float64
    assert connect1.shape == (nx * ny, 8)
    assert connect1.dtype == np.int64
    expected_nnodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    assert coords1.shape[0] == expected_nnodes
    assert connect1.shape[0] == nx * ny
    assert np.allclose(coords1, coords2)
    assert np.array_equal(connect1, connect2)
    x_coords = coords1[:, 0]
    y_coords = coords1[:, 1]
    assert np.any(np.isclose(x_coords, xl))
    assert np.any(np.isclose(x_coords, xh))
    assert np.any(np.isclose(y_coords, yl))
    assert np.any(np.isclose(y_coords, yh))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy
    x_normalized = np.round((x_coords - xl) / half_dx) * half_dx + xl
    assert np.allclose(x_coords, x_normalized)
    y_normalized = np.round((y_coords - yl) / half_dy) * half_dy + yl
    assert np.allclose(y_coords, y_normalized)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < nnodes)
    for (elem_idx, elem) in enumerate(connect):
        assert len(set(elem)) == 8, f'Element {elem_idx} has repeated node indices'
    for (elem_idx, elem) in enumerate(connect):
        (n1, n2, n3, n4) = (elem[0], elem[1], elem[2], elem[3])
        p1 = coords[n1]
        p2 = coords[n2]
        p3 = coords[n3]
        p4 = coords[n4]
        area = 0.5 * abs((p2[0] - p1[0]) * (p4[1] - p1[1]) - (p4[0] - p1[0]) * (p2[1] - p1[1]) + (p3[0] - p2[0]) * (p1[1] - p2[1]) - (p1[0] - p2[0]) * (p3[1] - p2[1]))
        assert area > 0, f'Element {elem_idx} has non-positive area'
    for (elem_idx, elem) in enumerate(connect):
        (n1, n2, n3, n4) = (elem[0], elem[1], elem[2], elem[3])
        (n5, n6, n7, n8) = (elem[4], elem[5], elem[6], elem[7])
        (p1, p2, p3, p4) = (coords[n1], coords[n2], coords[n3], coords[n4])
        (p5, p6, p7, p8) = (coords[n5], coords[n6], coords[n7], coords[n8])
        assert np.allclose(p5, 0.5 * (p1 + p2))
        assert np.allclose(p6, 0.5 * (p2 + p3))
        assert np.allclose(p7, 0.5 * (p3 + p4))
        assert np.allclose(p8, 0.5 * (p4 + p1))
    edge_to_elems = {}
    for (elem_idx, elem) in enumerate(connect):
        (n1, n2, n3, n4) = (elem[0], elem[1], elem[2], elem[3])
        edge_key = tuple(sorted([n1, n2]))
        if edge_key not in edge_to_elems:
            edge_to_elems[edge_key] = []
        edge_to_elems[edge_key].append((elem_idx, 'bottom'))
        edge_key = tuple(sorted([n2, n3]))
        if edge_key not in edge_to_elems:
            edge_to_elems[edge_key] = []
        edge_to_elems[edge_key].append((elem_idx, 'right'))
        edge_key = tuple(sorted([n3, n4]))
        if edge_key not in edge_to_elems:
            edge_to_elems[edge_key] = []
        edge_to_elems[edge_key].append((elem_idx, 'top'))
        edge_key = tuple(sorted([n4, n1]))
        if edge_key not in edge_to_elems:
            edge_to_elems[edge_key] = []
        edge_to_elems[edge_key].append((elem_idx, 'left'))
    for (edge_key, elem_list) in edge_to_elems.items():
        if len(elem_list) > 1:
            for (e1_idx, e1_side) in elem_list:
                for (e2_idx, e2_side) in elem_list:
                    if e1_idx < e2_idx:
                        assert edge_key == edge_key, 'Shared edge node IDs must match'

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