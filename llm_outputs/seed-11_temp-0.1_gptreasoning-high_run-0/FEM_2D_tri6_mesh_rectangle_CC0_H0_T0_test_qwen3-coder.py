def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 1.0, 1.0)
    (nx, ny) = (2, 2)
    (coords1, connect1) = fcn(xl, yl, xh, yh, nx, ny)
    (coords2, connect2) = fcn(xl, yl, xh, yh, nx, ny)
    np.testing.assert_array_equal(coords1, coords2)
    np.testing.assert_array_equal(connect1, connect2)
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    nnodes = npx * npy
    nelems = 2 * nx * ny
    assert coords1.shape == (nnodes, 2)
    assert connect1.shape == (nelems, 6)
    assert coords1.dtype == np.float64
    assert connect1.dtype == np.int64
    expected_corners = np.array([[xl, yl], [xh, yl], [xl, yh], [xh, yh]])
    corner_indices = [0, nx * 2, ny * 2 * npx, ny * 2 * npx + nx * 2]
    computed_corners = coords1[corner_indices]
    np.testing.assert_array_almost_equal(computed_corners, expected_corners)
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x_coords = np.linspace(xl, xh, npx)
    y_coords = np.linspace(yl, yh, npy)
    (X, Y) = np.meshgrid(x_coords, y_coords, indexing='xy')
    expected_coords = np.column_stack((X.ravel(), Y.ravel()))
    np.testing.assert_array_almost_equal(coords1, expected_coords)

def test_tri6_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain.
    Checks:
    """
    (xl, yl, xh, yh) = (0.0, 0.0, 2.0, 1.0)
    (nx, ny) = (2, 2)
    (coords, connect) = fcn(xl, yl, xh, yh, nx, ny)
    nnodes = coords.shape[0]
    assert np.all(connect >= 0)
    assert np.all(connect < nnodes)
    for elem in connect:
        assert len(np.unique(elem)) == 6
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    for (i, (cy, cx)) in enumerate([(cy, cx) for cy in range(ny) for cx in range(nx)]):
        (base_ix, base_iy) = (2 * cx, 2 * cy)
        base_id = base_iy * npx + base_ix
        br = base_id + 2
        tl = base_id + npx * 2
        bl = base_id
        elem1 = connect[2 * (cy * nx + cx)]
        np.testing.assert_array_equal(elem1[:3], [br, tl, bl])
        p_br = coords[br]
        p_tl = coords[tl]
        p_bl = coords[bl]
        cross_product = (p_tl[0] - p_br[0]) * (p_bl[1] - p_br[1]) - (p_tl[1] - p_br[1]) * (p_bl[0] - p_br[0])
        assert cross_product > 0
        n4 = elem1[3]
        n5 = elem1[4]
        n6 = elem1[5]
        np.testing.assert_array_almost_equal(coords[n4], 0.5 * (p_br + p_tl))
        np.testing.assert_array_almost_equal(coords[n5], 0.5 * (p_tl + p_bl))
        np.testing.assert_array_almost_equal(coords[n6], 0.5 * (p_bl + p_br))
        tr = base_id + npx * 2 + 2
        elem2 = connect[2 * (cy * nx + cx) + 1]
        np.testing.assert_array_equal(elem2[:3], [tr, tl, br])
        p_tr = coords[tr]
        cross_product = (p_tl[0] - p_tr[0]) * (p_br[1] - p_tr[1]) - (p_tl[1] - p_tr[1]) * (p_br[0] - p_tr[0])
        assert cross_product > 0
        n4 = elem2[3]
        n5 = elem2[4]
        n6 = elem2[5]
        np.testing.assert_array_almost_equal(coords[n4], 0.5 * (p_tr + p_tl))
        np.testing.assert_array_almost_equal(coords[n5], 0.5 * (p_tl + p_br))
        np.testing.assert_array_almost_equal(coords[n6], 0.5 * (p_br + p_tr))
    for cy in range(ny):
        for cx in range(1, nx):
            idx1 = 2 * (cy * nx + cx)
            idx2 = 2 * (cy * nx + cx - 1) + 1
            elem1 = connect[idx1]
            elem2 = connect[idx2]
            edge1 = {elem1[1], elem1[2]}
            edge2 = {elem2[0], elem2[2]}
            assert edge1 == edge2

def test_tri6_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs.
    Checks:
    """
    valid_args = [0.0, 0.0, 1.0, 1.0, 2, 2]
    args = valid_args.copy()
    args[4] = 0
    with pytest.raises(ValueError):
        fcn(*args)
    args[4] = -1
    with pytest.raises(ValueError):
        fcn(*args)
    args = valid_args.copy()
    args[5] = 0
    with pytest.raises(ValueError):
        fcn(*args)
    args[5] = -1
    with pytest.raises(ValueError):
        fcn(*args)
    args = valid_args.copy()
    args[0] = 1.0
    args[2] = 1.0
    with pytest.raises(ValueError):
        fcn(*args)
    args[0] = 2.0
    with pytest.raises(ValueError):
        fcn(*args)
    args = valid_args.copy()
    args[1] = 1.0
    args[3] = 1.0
    with pytest.raises(ValueError):
        fcn(*args)
    args[1] = 2.0
    with pytest.raises(ValueError):
        fcn(*args)