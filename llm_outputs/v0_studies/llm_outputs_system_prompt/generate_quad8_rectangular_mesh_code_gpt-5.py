def generate_quad8_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    import numpy as np
    if not (nx >= 1 and ny >= 1):
        raise ValueError('nx and ny must be >= 1')
    if not (xl < xh and yl < yh):
        raise ValueError('Invalid domain bounds: require xl < xh and yl < yh')
    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    ix = np.arange(npx, dtype=np.int64)
    iy = np.arange(npy, dtype=np.int64)
    (X_idx, Y_idx) = np.meshgrid(ix, iy, indexing='xy')
    mask_keep = ~((X_idx % 2 == 1) & (Y_idx % 2 == 1))
    grid_id = np.full((npy, npx), -1, dtype=np.int64)
    flat_mask = mask_keep.ravel(order='C')
    grid_id_flat = grid_id.ravel(order='C')
    count = int(np.count_nonzero(flat_mask))
    grid_id_flat[flat_mask] = np.arange(count, dtype=np.int64)
    step_x = 0.5 * dx
    step_y = 0.5 * dy
    x_axis = xl + step_x * ix.astype(np.float64)
    y_axis = yl + step_y * iy.astype(np.float64)
    (X, Y) = np.meshgrid(x_axis, y_axis, indexing='xy')
    coords = np.empty((count, 2), dtype=np.float64)
    coords[:, 0] = X.ravel(order='C')[flat_mask]
    coords[:, 1] = Y.ravel(order='C')[flat_mask]
    Ne = nx * ny
    connect = np.empty((Ne, 8), dtype=np.int64)
    e = 0
    for cy in range(ny):
        iy0 = 2 * cy
        for cx in range(nx):
            ix0 = 2 * cx
            n1 = grid_id[iy0, ix0]
            n2 = grid_id[iy0, ix0 + 2]
            n3 = grid_id[iy0 + 2, ix0 + 2]
            n4 = grid_id[iy0 + 2, ix0]
            n5 = grid_id[iy0, ix0 + 1]
            n6 = grid_id[iy0 + 1, ix0 + 2]
            n7 = grid_id[iy0 + 2, ix0 + 1]
            n8 = grid_id[iy0 + 1, ix0]
            connect[e, 0] = n1
            connect[e, 1] = n2
            connect[e, 2] = n3
            connect[e, 3] = n4
            connect[e, 4] = n5
            connect[e, 5] = n6
            connect[e, 6] = n7
            connect[e, 7] = n8
            e += 1
    return (coords, connect)