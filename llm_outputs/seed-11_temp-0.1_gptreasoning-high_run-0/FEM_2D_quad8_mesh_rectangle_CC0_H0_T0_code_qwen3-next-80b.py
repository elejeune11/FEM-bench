def FEM_2D_quad8_mesh_rectangle_CC0_H0_T0(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh')
    if nx < 1 or ny < 1:
        raise ValueError('Number of subdivisions nx and ny must be at least 1')
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    x_coords = np.linspace(xl, xh, npx)
    y_coords = np.linspace(yl, yh, npy)
    (X, Y) = np.meshgrid(x_coords, y_coords, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    ix = np.arange(npx)
    iy = np.arange(npy)
    (ix_grid, iy_grid) = np.meshgrid(ix, iy, indexing='xy')
    ix_flat = ix_grid.flatten()
    iy_flat = iy_grid.flatten()
    mask = ~((ix_flat % 2 == 1) & (iy_flat % 2 == 1))
    coords = np.column_stack((X_flat[mask], Y_flat[mask])).astype(np.float64)
    node_id_map = np.zeros((npx, npy), dtype=np.int64)
    next_id = 0
    for iy in range(npy):
        for ix in range(npx):
            if not (ix % 2 == 1 and iy % 2 == 1):
                node_id_map[ix, iy] = next_id
                next_id += 1
    Ne = nx * ny
    connect = np.zeros((Ne, 8), dtype=np.int64)
    for cy in range(ny):
        for cx in range(nx):
            cell_idx = cy * nx + cx
            ix0 = 2 * cx
            iy0 = 2 * cy
            N1 = node_id_map[ix0, iy0]
            N2 = node_id_map[ix0 + 2, iy0]
            N3 = node_id_map[ix0 + 2, iy0 + 2]
            N4 = node_id_map[ix0, iy0 + 2]
            N5 = node_id_map[ix0 + 1, iy0]
            N6 = node_id_map[ix0 + 2, iy0 + 1]
            N7 = node_id_map[ix0 + 1, iy0 + 2]
            N8 = node_id_map[ix0, iy0 + 1]
            connect[cell_idx] = [N1, N2, N3, N4, N5, N6, N7, N8]
    return (coords, connect)