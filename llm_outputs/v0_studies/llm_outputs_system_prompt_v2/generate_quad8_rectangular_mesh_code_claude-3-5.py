def generate_quad8_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be >= 1')
    if xl >= xh or yl >= yh:
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh')
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    x = np.linspace(xl, xh, npx)
    y = np.linspace(yl, yh, npy)
    (X, Y) = np.meshgrid(x, y, indexing='xy')
    (ix, iy) = np.meshgrid(np.arange(npx), np.arange(npy), indexing='xy')
    valid_mask = ~((ix % 2 == 1) & (iy % 2 == 1))
    coords = np.column_stack((X[valid_mask], Y[valid_mask]))
    node_map = np.full((npy, npx), -1, dtype=np.int64)
    node_map[valid_mask] = np.arange(np.sum(valid_mask))
    connect = np.empty((nx * ny, 8), dtype=np.int64)
    elem = 0
    for cy in range(ny):
        for cx in range(nx):
            (ix0, iy0) = (2 * cx, 2 * cy)
            n1 = node_map[iy0, ix0]
            n2 = node_map[iy0, ix0 + 2]
            n3 = node_map[iy0 + 2, ix0 + 2]
            n4 = node_map[iy0 + 2, ix0]
            n5 = node_map[iy0, ix0 + 1]
            n6 = node_map[iy0 + 1, ix0 + 2]
            n7 = node_map[iy0 + 2, ix0 + 1]
            n8 = node_map[iy0 + 1, ix0]
            connect[elem] = [n1, n2, n3, n4, n5, n6, n7, n8]
            elem += 1
    return (coords, connect)