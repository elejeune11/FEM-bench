def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    import numpy as np
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes_expected = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    Ne_expected = nx * ny
    assert coords1.shape == (Nnodes_expected, 2)
    assert connect1.shape == (Ne_expected, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    xs = coords1[:, 0]
    ys = coords1[:, 1]
    assert np.isclose(xs.min(), xl)
    assert np.isclose(xs.max(), xh)
    assert np.isclose(ys.min(), yl)
    assert np.isclose(ys.max(), yh)
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]], dtype=float)
    for c in corners:
        assert np.any(np.all(np.isclose(coords1, c, rtol=0.0, atol=0.0), axis=1))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    unique_x = np.unique(xs)
    unique_y = np.unique(ys)
    assert unique_x.size == 2 * nx + 1
    assert unique_y.size == 2 * ny + 1
    assert np.allclose(np.diff(unique_x), 0.5 * dx)
    assert np.allclose(np.diff(unique_y), 0.5 * dy)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords1, coords2)
    assert np.array_equal(connect1, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    import numpy as np
    (xl, yl, xh, yh) = (-2.0, 1.5, 3.0, 6.5)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert connect.min() >= 0
    assert connect.max() < coords.shape[0]
    for row in connect:
        assert np.unique(row).size == 8

    def signed_area(pts):
        x = pts[:, 0]
        y = pts[:, 1]
        return 0.5 * (x[0] * y[1] - x[1] * y[0] + x[1] * y[2] - x[2] * y[1] + x[2] * y[3] - x[3] * y[2] + x[3] * y[0] - x[0] * y[3])
    for e in range(connect.shape[0]):
        idx = connect[e, [0, 1, 2, 3]]
        area = signed_area(coords[idx])
        assert area > 0.0
    for e in range(connect.shape[0]):
        (N1, N2, N3, N4, N5, N6, N7, N8) = connect[e]
        (c1, c2, c3, c4) = (coords[N1], coords[N2], coords[N3], coords[N4])
        (c5, c6, c7, c8) = (coords[N5], coords[N6], coords[N7], coords[N8])
        assert np.allclose(c5, 0.5 * (c1 + c2))
        assert np.allclose(c6, 0.5 * (c2 + c3))
        assert np.allclose(c7, 0.5 * (c3 + c4))
        assert np.allclose(c8, 0.5 * (c4 + c1))
    for cy in range(ny):
        for cx in range(nx - 1):
            e_left = cy * nx + cx
            e_right = cy * nx + (cx + 1)
            assert connect[e_left, 1] == connect[e_right, 0]
            assert connect[e_left, 2] == connect[e_right, 3]
    for cy in range(ny - 1):
        for cx in range(nx):
            e_bottom = cy * nx + cx
            e_top = (cy + 1) * nx + cx
            assert connect[e_bottom, 2] == connect[e_top, 1]
            assert connect[e_bottom, 3] == connect[e_top, 0]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    import pytest
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -3, 2)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 2, -1)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 2.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 2.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 2.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 3.0, 1.0, 2.0, 1, 1)