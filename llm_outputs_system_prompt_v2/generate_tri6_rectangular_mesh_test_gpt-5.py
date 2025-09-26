def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    expected_nodes = npx * npy
    expected_elements = 2 * nx * ny
    assert isinstance(coords, np.ndarray) and isinstance(connect, np.ndarray)
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elements, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    idx_BL = 0 * npx + 0
    idx_BR = 0 * npx + (npx - 1)
    idx_TL = (npy - 1) * npx + 0
    idx_TR = (npy - 1) * npx + (npx - 1)
    assert np.allclose(coords[idx_BL], [xl, yl])
    assert np.allclose(coords[idx_BR], [xh, yl])
    assert np.allclose(coords[idx_TL], [xl, yh])
    assert np.allclose(coords[idx_TR], [xh, yh])
    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])
    assert np.allclose(x_unique, np.linspace(xl, xh, npx))
    assert np.allclose(y_unique, np.linspace(yl, yh, npy))
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (-1.0, 2.0, 3.0, 5.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < n_nodes
    for row in connect:
        assert np.unique(row).size == 6
    for row in connect:
        (n1, n2, n3) = (row[0], row[1], row[2])
        (x1, y1) = coords[n1]
        (x2, y2) = coords[n2]
        (x3, y3) = coords[n3]
        signed_area = 0.5 * ((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        assert signed_area > 0.0
    for row in connect:
        (n1, n2, n3, n4, n5, n6) = row
        mid12 = 0.5 * (coords[n1] + coords[n2])
        mid23 = 0.5 * (coords[n2] + coords[n3])
        mid31 = 0.5 * (coords[n3] + coords[n1])
        assert np.allclose(coords[n4], mid12, rtol=1e-14, atol=1e-14)
        assert np.allclose(coords[n5], mid23, rtol=1e-14, atol=1e-14)
        assert np.allclose(coords[n6], mid31, rtol=1e-14, atol=1e-14)
    edges_mids = {}
    for row in connect:
        (n1, n2, n3, n4, n5, n6) = row
        edges = [((n1, n2), n4), ((n2, n3), n5), ((n3, n1), n6)]
        for ((a, b), m) in edges:
            key = frozenset((a, b))
            edges_mids.setdefault(key, set()).add(m)
    for mids in edges_mids.values():
        assert len(mids) == 1

def test_tri6_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs.
    Checks:
    """
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, -2, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, -3)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, 0.0, -1.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 2.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 3.0, 1.0, -5.0, 1, 1)