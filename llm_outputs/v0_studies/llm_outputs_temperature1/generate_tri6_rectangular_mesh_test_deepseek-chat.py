def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    expected_nodes = npx * npy
    expected_elements = 2 * nx * ny
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elements, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.isclose(coords[0, 0], xl) and np.isclose(coords[0, 1], yl)
    assert np.isclose(coords[npx - 1, 0], xh) and np.isclose(coords[npx - 1, 1], yl)
    assert np.isclose(coords[-npx, 0], xl) and np.isclose(coords[-npx, 1], yh)
    assert np.isclose(coords[-1, 0], xh) and np.isclose(coords[-1, 1], yh)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    for i in range(npx):
        for j in range(npy):
            idx = j * npx + i
            expected_x = xl + 0.5 * dx * i
            expected_y = yl + 0.5 * dy * j
            assert np.isclose(coords[idx, 0], expected_x)
            assert np.isclose(coords[idx, 1], expected_y)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < n_nodes)
    for elem in connect:
        assert len(np.unique(elem)) == 6
    for elem in connect:
        (n1, n2, n3, n4, n5, n6) = elem
        assert np.allclose(coords[n4], (coords[n1] + coords[n2]) / 2)
        assert np.allclose(coords[n5], (coords[n2] + coords[n3]) / 2)
        assert np.allclose(coords[n6], (coords[n3] + coords[n1]) / 2)
        v1 = coords[n2] - coords[n1]
        v2 = coords[n3] - coords[n1]
        cross_z = v1[0] * v2[1] - v1[1] * v2[0]
        assert cross_z > 0
    npx = 2 * nx + 1
    for cy in range(ny):
        for cx in range(nx):
            cell_idx = cy * nx + cx
            elem1_idx = 2 * cell_idx
            elem2_idx = 2 * cell_idx + 1
            tri1 = connect[elem1_idx]
            tri2 = connect[elem2_idx]
            shared_nodes = set(tri1) & set(tri2)
            assert len(shared_nodes) >= 3

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
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