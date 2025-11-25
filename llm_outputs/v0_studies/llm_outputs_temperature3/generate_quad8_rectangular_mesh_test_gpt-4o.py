def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements."""
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    expected_nodes = (2 * nx + 1) * (2 * ny + 1) - nx * ny
    expected_elements = nx * ny
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elements, 8)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.allclose(coords[0], [xl, yl])
    assert np.allclose(coords[2], [xh, yl])
    assert np.allclose(coords[6], [xh, yh])
    assert np.allclose(coords[4], [xl, yh])
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    expected_coords = np.array([[xl + i * 0.5 * dx, yl + j * 0.5 * dy] for j in range(2 * ny + 1) for i in range(2 * nx + 1) if not (i % 2 == 1 and j % 2 == 1)])
    assert np.allclose(coords, expected_coords)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_quad8_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain for Quad8 elements."""
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (2, 1)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    for element in connect:
        assert np.all(element >= 0)
        assert np.all(element < len(coords))
        assert len(set(element)) == 8
    for element in connect:
        (N1, N2, N3, N4) = coords[element[:4]]
        area = 0.5 * ((N2[0] - N1[0]) * (N2[1] + N1[1]) + (N3[0] - N2[0]) * (N3[1] + N2[1]) + (N4[0] - N3[0]) * (N4[1] + N3[1]) + (N1[0] - N4[0]) * (N1[1] + N4[1]))
        assert area > 0
    for element in connect:
        (N1, N2, N3, N4, N5, N6, N7, N8) = coords[element]
        assert np.allclose(N5, (N1 + N2) / 2)
        assert np.allclose(N6, (N2 + N3) / 2)
        assert np.allclose(N7, (N3 + N4) / 2)
        assert np.allclose(N8, (N4 + N1) / 2)
    shared_edges = set()
    for element in connect:
        edges = {(element[i], element[(i + 1) % 4]) for i in range(4)}
        for edge in edges:
            if edge in shared_edges:
                shared_edges.remove(edge)
            else:
                shared_edges.add(edge)
    assert len(shared_edges) == 0

def test_quad8_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs in Quad8 mesh generation."""
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 1.0, 1.0, 0.0, 1, 1)