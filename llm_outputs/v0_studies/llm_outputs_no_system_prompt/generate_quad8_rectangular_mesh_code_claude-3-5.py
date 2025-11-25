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
    mask = ~((np.arange(npx).reshape(1, -1) % 2 == 1) & (np.arange(npy).reshape(-1, 1) % 2 == 1))
    coords = np.column_stack((X[mask], Y[mask])).astype(np.float64)
    node_ids = np.full((npy, npx), -1)
    node_ids[mask] = np.arange(np.sum(mask))
    connect = np.zeros((nx * ny, 8), dtype=np.int64)
    for cy in range(ny):
        for cx in range(nx):
            (ix0, iy0) = (2 * cx, 2 * cy)
            elem_idx = cy * nx + cx
            connect[elem_idx, 0] = node_ids[iy0, ix0]
            connect[elem_idx, 1] = node_ids[iy0, ix0 + 2]
            connect[elem_idx, 2] = node_ids[iy0 + 2, ix0 + 2]
            connect[elem_idx, 3] = node_ids[iy0 + 2, ix0]
            connect[elem_idx, 4] = node_ids[iy0, ix0 + 1]
            connect[elem_idx, 5] = node_ids[iy0 + 1, ix0 + 2]
            connect[elem_idx, 6] = node_ids[iy0 + 2, ix0 + 1]
            connect[elem_idx, 7] = node_ids[iy0 + 1, ix0]
    return (coords, connect)