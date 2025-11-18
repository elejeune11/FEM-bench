def generate_tri6_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('Invalid domain bounds: xl must be < xh and yl must be < yh.')
    if nx < 1 or ny < 1:
        raise ValueError('Invalid subdivisions: nx and ny must be >= 1.')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x = np.linspace(xl, xh, npx)
    y = np.linspace(yl, yh, npy)
    (xv, yv) = np.meshgrid(x, y, indexing='xy')
    coords = np.vstack((xv.flatten(), yv.flatten())).T
    connect = np.zeros((2 * nx * ny, 6), dtype=np.int64)
    element_idx = 0
    for cy in range(ny):
        for cx in range(nx):
            bl = 2 * cy * npx + 2 * cx
            br = bl + 2
            tl = bl + 2 * npx
            tr = tl + 2
            mbltr = bl + npx + 1
            mblbr = bl + 1
            mtltr = tl + 1
            connect[element_idx] = [br, tl, bl, mbltr, mblbr, br + npx + 1]
            element_idx += 1
            connect[element_idx] = [tr, tl, br, mtltr, mbltr, tr - npx - 1]
            element_idx += 1
    return (coords, connect)