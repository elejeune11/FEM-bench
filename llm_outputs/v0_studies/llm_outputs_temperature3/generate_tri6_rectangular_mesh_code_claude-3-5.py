def generate_tri6_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be >= 1')
    if xl >= xh or yl >= yh:
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x = np.linspace(xl, xh, npx)
    y = np.linspace(yl, yh, npy)
    (X, Y) = np.meshgrid(x, y, indexing='xy')
    coords = np.column_stack((X.ravel(), Y.ravel()))
    connect = np.zeros((2 * nx * ny, 6), dtype=np.int64)
    elem = 0
    for cy in range(ny):
        for cx in range(nx):
            sw = 2 * cy * npx + 2 * cx
            se = sw + 2
            nw = sw + 2 * npx
            ne = nw + 2
            connect[elem] = [se, nw, sw, se + npx, sw + npx, se - 1]
            elem += 1
            connect[elem] = [ne, nw, se, ne - 1, se + npx, ne - npx]
            elem += 1
    return (coords, connect)