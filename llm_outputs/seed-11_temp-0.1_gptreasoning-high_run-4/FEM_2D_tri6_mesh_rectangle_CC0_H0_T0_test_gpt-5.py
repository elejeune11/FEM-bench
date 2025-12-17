def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    xl, yl, xh, yh = (0.0, 0.0, 1.0, 1.0)
    nx, ny = (2, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    npx, npy = (2 * nx + 1, 2 * ny + 1)
    expected_nodes = npx * npy
    expected_elements = 2 * nx * ny
    assert coords.shape == (expected_nodes, 2)
    assert connect.shape == (expected_elements, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    stepx = 0.5 * dx
    stepy = 0.5 * dy
    x_unique = np.unique(coords[:, 0])
    y_unique = np.unique(coords[:, 1])
    expected_x = xl + stepx * np.arange(npx)
    expected_y = yl + stepy * np.arange(npy)
    assert np.allclose(x_unique, expected_x)
    assert np.allclose(y_unique, expected_y)
    bl = 0
    br = npx - 1
    tl = (npy - 1) * npx
    tr = npx * npy - 1
    assert np.allclose(coords[bl], [xl, yl])
    assert np.allclose(coords[br], [xh, yl])
    assert np.allclose(coords[tl], [xl, yh])
    assert np.allclose(coords[tr], [xh, yh])
    coords2, connect2 = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(coords, coords2)
    assert np.array_equal(connect, connect2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    xl, yl, xh, yh = (-1.5, 2.0, 2.25, 5.5)
    nx, ny = (3, 2)
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)
    Nnodes = coords.shape[0]
    assert connect.min() >= 0
    assert connect.max() < Nnodes
    uniq_counts = np.array([np.unique(row).size for row in connect])
    assert np.all(uniq_counts == 6)
    A = coords[connect[:, 0]]
    B = coords[connect[:, 1]]
    C = coords[connect[:, 2]]
    signed_area = (B[:, 0] - A[:, 0]) * (C[:, 1] - A[:, 1]) - (B[:, 1] - A[:, 1]) * (C[:, 0] - A[:, 0])
    assert np.all(signed_area > 0.0)
    M4 = coords[connect[:, 3]]
    M5 = coords[connect[:, 4]]
    M6 = coords[connect[:, 5]]
    assert np.allclose(M4, 0.5 * (A + B))
    assert np.allclose(M5, 0.5 * (B + C))
    assert np.allclose(M6, 0.5 * (C + A))
    edges = []
    for row in connect:
        e12 = frozenset((row[0], row[1], row[3]))
        e23 = frozenset((row[1], row[2], row[4]))
        e31 = frozenset((row[2], row[0], row[5]))
        edges.extend([e12, e23, e31])
    counts = Counter(edges)
    multiplicities = list(counts.values())
    assert set(multiplicities).issubset({1, 2})
    expected_interior_edges = nx * ny + (nx - 1) * ny + nx * (ny - 1)
    assert sum((1 for v in multiplicities if v == 2)) == expected_interior_edges
    assert sum((1 for v in multiplicities if v == 1)) > 0

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    xl, yl, xh, yh = (0.0, 0.0, 1.0, 1.0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 0, 1)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, -2, 1)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 1, 0)
    with pytest.raises(ValueError):
        fcn(xl, yl, xh, yh, 1, -3)
    with pytest.raises(ValueError):
        fcn(1.0, yl, 1.0, yh, 1, 1)
    with pytest.raises(ValueError):
        fcn(2.0, yl, 1.0, yh, 1, 1)
    with pytest.raises(ValueError):
        fcn(xl, 1.0, xh, 1.0, 1, 1)
    with pytest.raises(ValueError):
        fcn(xl, 3.0, xh, 2.0, 1, 1)