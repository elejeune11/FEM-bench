def FEM_2D_quad8_mesh_rectangle_CC0_H0_T0(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('xl < xh and yl < yh required')
    if nx < 1 or ny < 1:
        raise ValueError('nx ≥ 1 and ny ≥ 1 required')
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    x_refined = np.linspace(xl, xh, npx)
    y_refined = np.linspace(yl, yh, npy)
    (X, Y) = np.meshgrid(x_refined, y_refined, indexing='xy')
    X_flat = X.flatten()
    Y_flat = Y.flatten()
    (iy_all, ix_all) = np.meshgrid(np.arange(npy), np.arange(npx), indexing='ij')
    iy_flat = iy_all.flatten()
    ix_flat = ix_all.flatten()
    mask = ~((ix_flat % 2 == 1) & (iy_flat % 2 == 1))
    coords = np.column_stack((X_flat[mask], Y_flat[mask])).astype(np.float64)
    node_id_grid = np.full((npy, npx), -1, dtype=np.int64)
    valid_indices = np.where(mask)[0]
    node_ids = np.arange(len(valid_indices), dtype=np.int64)
    iy_valid = iy_flat[mask]
    ix_valid = ix_flat[mask]
    node_id_grid[iy_valid, ix_valid] = node_ids
    connect = []
    for cy in range(ny):
        for cx in range(nx):
            ix0 = 2 * cx
            iy0 = 2 * cy
            N1 = (ix0, iy0)
            N2 = (ix0 + 2, iy0)
            N3 = (ix0 + 2, iy0 + 2)
            N4 = (ix0, iy0 + 2)
            N5 = (ix0 + 1, iy0)
            N6 = (ix0 + 2, iy0 + 1)
            N7 = (ix0 + 1, iy0 + 2)
            N8 = (ix0, iy0 + 1)
            n1 = node_id_grid[N1[1], N1[0]]
            n2 = node_id_grid[N2[1], N2[0]]
            n3 = node_id_grid[N3[1], N3[0]]
            n4 = node_id_grid[N4[1], N4[0]]
            n5 = node_id_grid[N5[1], N5[0]]
            n6 = node_id_grid[N6[1], N6[0]]
            n7 = node_id_grid[N7[1], N7[0]]
            n8 = node_id_grid[N8[1], N8[0]]
            connect.append([n1, n2, n3, n4, n5, n6, n7, n8])
    connect = np.array(connect, dtype=np.int64)
    return (coords, connect)