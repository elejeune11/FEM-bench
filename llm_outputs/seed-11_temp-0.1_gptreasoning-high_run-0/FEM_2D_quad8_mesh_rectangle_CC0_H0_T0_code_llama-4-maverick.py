def FEM_2D_quad8_mesh_rectangle_CC0_H0_T0(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('xl < xh and yl < yh must hold')
    if nx < 1 or ny < 1:
        raise ValueError('nx >= 1 and ny >= 1 must hold')
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    Nnodes = npx * npy - nx * ny
    x_refined = np.linspace(xl, xh, npx)
    y_refined = np.linspace(yl, yh, npy)
    (x_grid, y_grid) = np.meshgrid(x_refined, y_refined, indexing='xy')
    mask = ~(np.arange(npx)[:, None] % 2 == 1 & np.arange(npy)[None, :] % 2 == 1)
    coords = np.column_stack((x_grid.flatten(), y_grid.flatten()))[mask.flatten()]
    connect = np.zeros((nx * ny, 8), dtype=np.int64)
    node_id = 0
    node_ids = np.full((npy, npx), -1, dtype=np.int64)
    for iy in range(npy):
        for ix in range(npx):
            if not (ix % 2 == 1 and iy % 2 == 1):
                node_ids[iy, ix] = node_id
                node_id += 1
    for cy in range(ny):
        for cx in range(nx):
            (ix0, iy0) = (2 * cx, 2 * cy)
            element_id = cy * nx + cx
            connect[element_id] = [node_ids[iy0, ix0], node_ids[iy0, ix0 + 2], node_ids[iy0 + 2, ix0 + 2], node_ids[iy0 + 2, ix0], node_ids[iy0, ix0 + 1], node_ids[iy0 + 1, ix0 + 2], node_ids[iy0 + 2, ix0 + 1], node_ids[iy0 + 1, ix0]]
    return (coords, connect)