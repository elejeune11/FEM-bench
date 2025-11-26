def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords.shape == (21, 2)
    assert connect.shape == (4, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.isclose(coords[:, 0].min(), xl)
    assert np.isclose(coords[:, 0].max(), xh)
    assert np.isclose(coords[:, 1].min(), yl)
    assert np.isclose(coords[:, 1].max(), yh)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    step_x = dx / 2.0
    step_y = dy / 2.0
    k_x = (coords[:, 0] - xl) / step_x
    k_y = (coords[:, 1] - yl) / step_y
    assert np.allclose(k_x, np.round(k_x))
    assert np.allclose(k_y, np.round(k_y))
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    np.testing.assert_array_equal(coords, coords2)
    np.testing.assert_array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (2, 1)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    num_nodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < num_nodes
    for elem in connect:
        assert len(np.unique(elem)) == 8
    for elem_indices in connect:
        pts = coords[elem_indices]
        (n1, n2, n3, n4) = (pts[0], pts[1], pts[2], pts[3])
        (n5, n6, n7, n8) = (pts[4], pts[5], pts[6], pts[7])
        corners = np.array([n1, n2, n3, n4])
        x = corners[:, 0]
        y = corners[:, 1]
        area = 0.5 * np.sum(x * np.roll(y, -1) - np.roll(x, -1) * y)
        assert area > 0
        assert np.allclose(n5, (n1 + n2) / 2.0)
        assert np.allclose(n6, (n2 + n3) / 2.0)
        assert np.allclose(n7, (n3 + n4) / 2.0)
        assert np.allclose(n8, (n4 + n1) / 2.0)
    el0 = connect[0]
    el1 = connect[1]
    assert el0[1] == el1[0]
    assert el0[2] == el1[3]
    assert el0[5] == el1[7]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
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
        fcn(1.0, yl, 0.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(0.0, yl, 0.0, yh, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 1.0, xh, 0.0, nx, ny)
    with pytest.raises(ValueError):
        fcn(xl, 0.0, xh, 0.0, nx, ny)