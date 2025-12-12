def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements."""
    import numpy as np
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    Nnodes_expected = npx * npy - nx * ny
    Ne_expected = nx * ny
    assert coords.shape == (Nnodes_expected, 2)
    assert connect.shape == (Ne_expected, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    corners = np.array([[xl, yl], [xh, yl], [xh, yh], [xl, yh]], dtype=np.float64)
    for c in corners:
        assert any((np.allclose(c, p) for p in coords))
    step_x = dx / 2.0
    step_y = dy / 2.0
    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    assert np.allclose(unique_x, np.linspace(xl, xh, npx))
    assert np.allclose(unique_y, np.linspace(yl, yh, npy))
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements."""
    import numpy as np
    (xl, yl, xh, yh) = (-1.0, 0.0, 2.0, 1.5)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    Ne = connect.shape[0]
    assert connect.min() >= 0
    assert connect.max() <= Nnodes - 1
    for row in connect:
        assert len(set((int(i) for i in row))) == 8
    for row in connect:
        (N1, N2, N3, N4) = row[0:4]
        poly = coords[[N1, N2, N3, N4], :]
        x = poly[:, 0]
        y = poly[:, 1]
        shoelace = 0.5 * (np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
        assert shoelace > 0.0
    for row in connect:
        (N1, N2, N3, N4, N5, N6, N7, N8) = row
        assert np.allclose(coords[N5], 0.5 * (coords[N1] + coords[N2]))
        assert np.allclose(coords[N6], 0.5 * (coords[N2] + coords[N3]))
        assert np.allclose(coords[N7], 0.5 * (coords[N3] + coords[N4]))
        assert np.allclose(coords[N8], 0.5 * (coords[N4] + coords[N1]))
    for cy in range(ny):
        for cx in range(nx):
            e = cy * nx + cx
            if cx < nx - 1:
                e_r = e + 1
                assert connect[e, 1] == connect[e_r, 0]
                assert connect[e, 5] == connect[e_r, 7]
            if cy < ny - 1:
                e_t = e + nx
                assert connect[e, 3] == connect[e_t, 0]
                assert connect[e, 2] == connect[e_t, 1]
                assert connect[e, 6] == connect[e_t, 4]

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation."""
    import pytest
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 1, 1)