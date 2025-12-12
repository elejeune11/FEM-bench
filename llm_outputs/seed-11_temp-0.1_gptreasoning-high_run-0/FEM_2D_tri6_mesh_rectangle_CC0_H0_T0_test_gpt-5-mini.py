def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    import numpy as np
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    Nnodes = npx * npy
    Ne = 2 * nx * ny
    assert isinstance(coords, np.ndarray)
    assert isinstance(connect, np.ndarray)
    assert coords.shape == (Nnodes, 2)
    assert connect.shape == (Ne, 6)
    assert coords.dtype == np.float64
    assert connect.dtype == np.int64
    assert np.allclose(coords[0], [xl, yl])
    assert np.allclose(coords[npx - 1], [xh, yl])
    assert np.allclose(coords[(npy - 1) * npx], [xl, yh])
    assert np.allclose(coords[(npy - 1) * npx + (npx - 1)], [xh, yh])
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    expected = np.empty((Nnodes, 2), dtype=np.float64)
    for iy in range(npy):
        for ix in range(npx):
            idx = iy * npx + ix
            expected[idx, 0] = xl + 0.5 * dx * ix
            expected[idx, 1] = yl + 0.5 * dy * iy
    assert np.allclose(coords, expected)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    assert np.array_equal(connect, connect2)
    assert np.allclose(coords, coords2)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    import numpy as np
    (xl, yl, xh, yh) = (0.5, -1.0, 2.5, 0.5)
    (nx, ny) = (3, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    Nnodes = npx * npy
    assert connect.shape[1] == 6
    for row in connect:
        row = np.asarray(row, dtype=int)
        assert row.min() >= 0
        assert row.max() < Nnodes
        assert len(set((int(x) for x in row.tolist()))) == 6
        (n1, n2, n3, n4, n5, n6) = row.tolist()
        c1 = coords[int(n1)]
        c2 = coords[int(n2)]
        c3 = coords[int(n3)]
        cross = (c2[0] - c1[0]) * (c3[1] - c1[1]) - (c3[0] - c1[0]) * (c2[1] - c1[1])
        assert cross > 0
        assert np.allclose(coords[int(n4)], 0.5 * (c1 + c2))
        assert np.allclose(coords[int(n5)], 0.5 * (c2 + c3))
        assert np.allclose(coords[int(n6)], 0.5 * (c3 + c1))
    edge_mids = {}
    for row in connect:
        (n1, n2, n3, n4, n5, n6) = map(int, row.tolist())
        edges = [((n1, n2), n4), ((n2, n3), n5), ((n3, n1), n6)]
        for ((a, b), m) in edges:
            key = tuple(sorted((int(a), int(b))))
            if key not in edge_mids:
                edge_mids[key] = set()
            edge_mids[key].add(int(m))
    for mids in edge_mids.values():
        assert len(mids) == 1

def test_tri6_mesh_invalid_inputs(fcn):
    """Validate error handling for invalid inputs.
    Checks:
    """
    try:
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    except ValueError:
        pass
    else:
        assert False, 'Expected ValueError when nx <= 0'
    try:
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)
    except ValueError:
        pass
    else:
        assert False, 'Expected ValueError when ny <= 0'
    try:
        fcn(1.0, 0.0, 1.0, 1.0, 1, 1)
    except ValueError:
        pass
    else:
        assert False, 'Expected ValueError when xl >= xh'
    try:
        fcn(0.0, 1.0, 1.0, 1.0, 1, 1)
    except ValueError:
        pass
    else:
        assert False, 'Expected ValueError when yl >= yh'