def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    assert coords.shape[0] == expected_nodes
    assert coords.shape[1] == 2
    assert connect.shape[0] == nx * ny
    assert connect.shape[1] == 8
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.any(np.logical_and(coords[:, 0] == xl, coords[:, 1] == yl))
    assert np.any(np.logical_and(coords[:, 0] == xl, coords[:, 1] == yh))
    assert np.any(np.logical_and(coords[:, 0] == xh, coords[:, 1] == yl))
    assert np.any(np.logical_and(coords[:, 0] == xh, coords[:, 1] == yh))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    half_dx = 0.5 * dx
    half_dy = 0.5 * dy
    for coord in coords:
        x_steps = (coord[0] - xl) / half_dx
        y_steps = (coord[1] - yl) / half_dy
        assert np.abs(x_steps - np.round(x_steps)) < 1e-10
        assert np.abs(y_steps - np.round(y_steps)) < 1e-10
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, -2.0, 3.0, 1.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.all(connect >= 0)
    assert np.all(connect < coords.shape[0])
    for elem in connect:
        assert len(np.unique(elem)) == 8
    for elem in connect:
        corners = elem[:4]
        corner_coords = coords[corners]
        area = 0.0
        for i in range(4):
            j = (i + 1) % 4
            area += corner_coords[i, 0] * corner_coords[j, 1]
            area -= corner_coords[j, 0] * corner_coords[i, 1]
        area *= 0.5
        assert area > 0
    for elem in connect:
        (n1, n2, n3, n4, n5, n6, n7, n8) = elem
        expected_n5 = 0.5 * (coords[n1] + coords[n2])
        assert np.allclose(coords[n5], expected_n5)
        expected_n6 = 0.5 * (coords[n2] + coords[n3])
        assert np.allclose(coords[n6], expected_n6)
        expected_n7 = 0.5 * (coords[n3] + coords[n4])
        assert np.allclose(coords[n7], expected_n7)
        expected_n8 = 0.5 * (coords[n4] + coords[n1])
        assert np.allclose(coords[n8], expected_n8)
    for i in range(connect.shape[0]):
        for j in range(i + 1, connect.shape[0]):
            elem1 = connect[i]
            elem2 = connect[j]
            edges1 = [(elem1[0], elem1[1]), (elem1[1], elem1[2]), (elem1[2], elem1[3]), (elem1[3], elem1[0])]
            edges2 = [(elem2[0], elem2[1]), (elem2[1], elem2[2]), (elem2[2], elem2[3]), (elem2[3], elem2[0])]
            for e1 in edges1:
                for e2 in edges2:
                    if set(e1) == set(e2):
                        edge_idx1 = edges1.index(e1)
                        edge_idx2 = edges2.index(e2)
                        midside1 = elem1[4 + edge_idx1]
                        midside2 = elem2[4 + edge_idx2]
                        assert midside1 == midside2

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    try:
        fcn(0.0, 0.0, 1.0, 1.0, 0, 2)
        assert False, 'Should have raised ValueError for nx=0'
    except ValueError:
        pass
    try:
        fcn(0.0, 0.0, 1.0, 1.0, -1, 2)
        assert False, 'Should have raised ValueError for nx=-1'
    except ValueError:
        pass
    try:
        fcn(0.0, 0.0, 1.0, 1.0, 2, 0)
        assert False, 'Should have raised ValueError for ny=0'
    except ValueError:
        pass
    try:
        fcn(0.0, 0.0, 1.0, 1.0, 2, -1)
        assert False, 'Should have raised ValueError for ny=-1'
    except ValueError:
        pass
    try:
        fcn(1.0, 0.0, 1.0, 1.0, 2, 2)
        assert False, 'Should have raised ValueError for xl=xh'
    except ValueError:
        pass
    try:
        fcn(2.0, 0.0, 1.0, 1.0, 2, 2)
        assert False, 'Should have raised ValueError for xl>xh'
    except ValueError:
        pass
    try:
        fcn(0.0, 1.0, 1.0, 1.0, 2, 2)
        assert False, 'Should have raised ValueError for yl=yh'
    except ValueError:
        pass
    try:
        fcn(0.0, 2.0, 1.0, 1.0, 2, 2)
        assert False, 'Should have raised ValueError for yl>yh'
    except ValueError:
        pass