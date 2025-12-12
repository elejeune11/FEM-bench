def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.
    Checks:
    """
    import numpy as np
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    expected_nodes = npx * npy - nx * ny
    expected_elements = nx * ny
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elements, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]], dtype=np.float64)
    for c in corners:
        assert np.any(np.all(np.isclose(coords, c, rtol=0, atol=1e-14), axis=1))
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    step_x = dx / 2.0
    step_y = dy / 2.0
    xvals = np.unique(coords[:, 0])
    yvals = np.unique(coords[:, 1])
    assert len(xvals) == npx
    assert len(yvals) == npy
    expected_x = np.linspace(xl, xh, npx, dtype=np.float64)
    expected_y = np.linspace(yl, yh, npy, dtype=np.float64)
    assert np.allclose(xvals, expected_x, rtol=0, atol=1e-14)
    assert np.allclose(yvals, expected_y, rtol=0, atol=1e-14)
    if npx > 1:
        assert np.allclose(np.diff(xvals), step_x, rtol=0, atol=1e-14)
    if npy > 1:
        assert np.allclose(np.diff(yvals), step_y, rtol=0, atol=1e-14)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords2.dtype == coords.dtype and connect2.dtype == connect.dtype
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements.
    Checks:
    """
    import numpy as np
    (xl, yl, xh, yh) = (0.0, 10.0, 3.0, 15.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < n_nodes
    for row in connect:
        (n1, n2, n3, n4, n5, n6, n7, n8) = row.tolist()
        assert len(np.unique(row)) == 8
        poly = coords[[n1, n2, n3, n4], :]
        x = poly[:, 0]
        y = poly[:, 1]
        area2 = x[0] * y[1] + x[1] * y[2] + x[2] * y[3] + x[3] * y[0]
        assert area2 > 0
        atol = 1e-12
        assert np.allclose(coords[n5], 0.5 * (coords[n1] + coords[n2]), rtol=0, atol=atol)
        assert np.allclose(coords[n6], 0.5 * (coords[n2] + coords[n3]), rtol=0, atol=atol)
        assert np.allclose(coords[n7], 0.5 * (coords[n3] + coords[n4]), rtol=0, atol=atol)
        assert np.allclose(coords[n8], 0.5 * (coords[n4] + coords[n1]), rtol=0, atol=atol)
    for cy in range(ny):
        for cx in range(nx):
            e = cy * nx + cx
            row = connect[e]
            if cx < nx - 1:
                er = cy * nx + (cx + 1)
                row_r = connect[er]
                assert {row[1], row[2]} == {row_r[3], row_r[0]}
                assert row[5] == row_r[7]
            if cy < ny - 1:
                et = (cy + 1) * nx + cx
                row_t = connect[et]
                assert {row[2], row[3]} == {row_t[0], row_t[1]}
                assert row[6] == row_t[4]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation.
    Checks:
    """
    import pytest
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -2, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, -3)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)