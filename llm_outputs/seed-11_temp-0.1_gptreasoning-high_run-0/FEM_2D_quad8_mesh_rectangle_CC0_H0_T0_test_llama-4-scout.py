def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 2.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes_expected = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    n_elements_expected = nx * ny
    assert len(coords) == n_nodes_expected
    assert len(connect) == n_elements_expected
    assert coords.shape == (n_nodes_expected, 2)
    assert coords.dtype == np.float64
    assert connect.shape == (n_elements_expected, 8)
    assert connect.dtype == np.int64
    corner_nodes = [0, nx, nx * (2 * ny + 1) + ny * 2, (2 * ny + 1) * (2 * nx + 1) - ny * 2 - 1]
    boundary_tol = 1e-12
    assert np.allclose(coords[corner_nodes, 0], [xl, xh, xh, xl], atol=boundary_tol)
    assert np.allclose(coords[corner_nodes, 1], [yl, yl, yh, yh], atol=boundary_tol)
    (dx, dy) = ((xh - xl) / nx, (yh - yl) / ny)
    for i in range(len(coords)):
        (x, y) = coords[i]
        assert np.isclose(x, xl + i % (2 * nx + 1) * 0.5 * dx, atol=1e-12)
        assert np.isclose(y, yl + i // (2 * nx + 1) * 0.5 * dy, atol=1e-12)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 3.0, 2.0)
    (nx, ny) = (2, 3)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.all(connect >= 0)
    assert np.all(connect < len(coords))
    assert len(np.unique(connect)) == len(connect)
    for i in range(len(connect)):
        (n1, n2, n3, n4, _, _, _, _) = connect[i]
        (x1, y1) = coords[n1]
        (x2, y2) = coords[n2]
        (x3, y3) = coords[n3]
        (x4, y4) = coords[n4]
        area = 0.5 * np.abs(x1 * (y2 - y4) + x2 * (y3 - y1) + x3 * (y4 - y2) + x4 * (y1 - y3))
        assert area > 0
    for i in range(len(connect)):
        (n1, n2, n3, n4, n5, n6, n7, n8) = connect[i]
        (x1, y1) = coords[n1]
        (x2, y2) = coords[n2]
        (x3, y3) = coords[n3]
        (x4, y4) = coords[n4]
        assert np.allclose(coords[n5], [(x1 + x2) / 2, (y1 + y2) / 2])
        assert np.allclose(coords[n6], [(x2 + x3) / 2, (y2 + y3) / 2])
        assert np.allclose(coords[n7], [(x3 + x4) / 2, (y3 + y4) / 2])
        assert np.allclose(coords[n8], [(x4 + x1) / 2, (y4 + y1) / 2])
    for i in range(len(connect)):
        (n1, n2, _, _, _, _, _, _) = connect[i]
        for j in range(i + 1, len(connect)):
            (n1_j, n2_j, _, _, _, _, _, _) = connect[j]
            if n1 == n1_j:
                assert n2 == n2_j
            if n1 == n2_j:
                assert n2 == n1_j

def test_quad8_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 1, 1)