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
            sw = cy * npx * 2 + cx * 2
            se = sw + 2
            nw = sw + npx * 2
            ne = nw + 2
            connect[elem, 0] = se
            connect[elem, 1] = nw
            connect[elem, 2] = sw
            connect[elem, 3] = se + npx
            connect[elem, 4] = sw + npx
            connect[elem, 5] = se - 1
            elem += 1
            connect[elem, 0] = ne
            connect[elem, 1] = nw
            connect[elem, 2] = se
            connect[elem, 3] = ne - 1
            connect[elem, 4] = se + npx
            connect[elem, 5] = ne - npx
            elem += 1
    return (coords, connect)