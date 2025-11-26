def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2x2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    n_nodes_expected = 25
    n_elems_expected = 2 * nx * ny
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    assert coords.shape == (n_nodes_expected, 2)
    assert coords.dtype == np.float64
    assert connect.shape == (n_elems_expected, 6)
    assert np.issubdtype(connect.dtype, np.integer)
    assert np.isclose(coords[:, 0].min(), xl)
    assert np.isclose(coords[:, 0].max(), xh)
    assert np.isclose(coords[:, 1].min(), yl)
    assert np.isclose(coords[:, 1].max(), yh)
    unique_x = np.unique(coords[:, 0])
    unique_y = np.unique(coords[:, 1])
    expected_x = np.linspace(xl, xh, 2 * nx + 1)
    expected_y = np.linspace(yl, yh, 2 * ny + 1)
    assert np.allclose(unique_x, expected_x)
    assert np.allclose(unique_y, expected_y)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (nx, ny) = (1, 2)
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 4.0)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    n_nodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < n_nodes
    for elem in connect:
        assert len(np.unique(elem)) == 6
    p1 = coords[connect[:, 0]]
    p2 = coords[connect[:, 1]]
    p3 = coords[connect[:, 2]]
    v1 = p2 - p1
    v2 = p3 - p1
    cross_product = v1[:, 0] * v2[:, 1] - v1[:, 1] * v2[:, 0]
    assert np.all(cross_product > 0), 'Corner nodes must be oriented CCW'
    p4 = coords[connect[:, 3]]
    p5 = coords[connect[:, 4]]
    p6 = coords[connect[:, 5]]
    assert np.allclose(p4, 0.5 * (p1 + p2)), 'N4 should be midpoint of N1-N2'
    assert np.allclose(p5, 0.5 * (p2 + p3)), 'N5 should be midpoint of N2-N3'
    assert np.allclose(p6, 0.5 * (p3 + p1)), 'N6 should be midpoint of N3-N1'
    n_cells = nx * ny
    for i in range(n_cells):
        tri1_idx = 2 * i
        tri2_idx = 2 * i + 1
        node_mid_1 = connect[tri1_idx, 3]
        node_mid_2 = connect[tri2_idx, 4]
        assert node_mid_1 == node_mid_2, f'Diagonal midside mismatch in cell {i} (elements {tri1_idx}, {tri2_idx})'

def test_tri6_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs.
    Checks:
    """
    base_args = {'xl': 0, 'yl': 0, 'xh': 1, 'yh': 1, 'nx': 1, 'ny': 1}

    def check_error(**kwargs):
        args = base_args.copy()
        args.update(kwargs)
        with pytest.raises(ValueError):
            fcn(**args)
    check_error(nx=0)
    check_error(nx=-1)
    check_error(ny=0)
    check_error(ny=-10)
    check_error(xl=1.0, xh=1.0)
    check_error(xl=2.0, xh=1.0)
    check_error(yl=1.0, yh=1.0)
    check_error(yl=5.0, yh=1.0)