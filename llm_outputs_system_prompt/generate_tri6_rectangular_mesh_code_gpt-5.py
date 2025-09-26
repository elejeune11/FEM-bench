def generate_tri6_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if not (xl < xh and yl < yh):
        raise ValueError('Invalid domain bounds: require xl < xh and yl < yh.')
    if not (isinstance(nx, int) and isinstance(ny, int) and (nx >= 1) and (ny >= 1)):
        raise ValueError('Invalid subdivisions: require nx >= 1 and ny >= 1.')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    ix = np.arange(npx, dtype=np.int64)
    iy = np.arange(npy, dtype=np.int64)
    x = xl + 0.5 * dx * ix.astype(np.float64)
    y = yl + 0.5 * dy * iy.astype(np.float64)
    (X, Y) = np.meshgrid(x, y, indexing='xy')
    coords = np.empty((npx * npy, 2), dtype=np.float64)
    coords[:, 0] = X.ravel(order='C')
    coords[:, 1] = Y.ravel(order='C')
    ne = 2 * nx * ny
    connect = np.empty((ne, 6), dtype=np.int64)
    e = 0
    for cy in range(ny):
        y0 = 2 * cy
        y1 = y0 + 1
        y2 = y0 + 2
        row0 = y0 * npx
        row1 = y1 * npx
        row2 = y2 * npx
        for cx in range(nx):
            x0 = 2 * cx
            x1 = x0 + 1
            x2 = x0 + 2
            bl = row0 + x0
            br = row0 + x2
            tl = row2 + x0
            tr = row2 + x2
            center = row1 + x1
            left_mid = row1 + x0
            bottom_mid = row0 + x1
            top_mid = row2 + x1
            right_mid = row1 + x2
            connect[e, 0] = br
            connect[e, 1] = tl
            connect[e, 2] = bl
            connect[e, 3] = center
            connect[e, 4] = left_mid
            connect[e, 5] = bottom_mid
            e += 1
            connect[e, 0] = tr
            connect[e, 1] = tl
            connect[e, 2] = br
            connect[e, 3] = top_mid
            connect[e, 4] = center
            connect[e, 5] = right_mid
            e += 1
    return (coords, connect)