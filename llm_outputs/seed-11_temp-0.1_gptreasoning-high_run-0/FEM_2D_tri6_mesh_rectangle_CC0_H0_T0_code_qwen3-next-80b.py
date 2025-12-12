def FEM_2D_tri6_mesh_rectangle_CC0_H0_T0(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh')
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be at least 1')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    ix_grid = np.arange(npx, dtype=np.float64)
    iy_grid = np.arange(npy, dtype=np.float64)
    x_coords = xl + 0.5 * dx * ix_grid
    y_coords = yl + 0.5 * dy * iy_grid
    (coords_x, coords_y) = np.meshgrid(x_coords, y_coords, indexing='xy')
    coords = np.column_stack((coords_x.flatten(order='C'), coords_y.flatten(order='C')))
    connect = np.zeros((2 * nx * ny, 6), dtype=np.int64)
    elem_idx = 0
    for cy in range(ny):
        for cx in range(nx):
            bl = cy * npx * 2 + cx * 2
            br = cy * npx * 2 + cx * 2 + 2
            tl = (cy * 2 + 2) * npx + cx * 2
            tr = (cy * 2 + 2) * npx + cx * 2 + 2
            m1 = (cy * 2 + 1) * npx + (cx * 2 + 1)
            m2 = (cy * 2 + 1) * npx + cx * 2
            m3 = cy * npx * 2 + cx * 2 + 1
            m4 = (cy * 2 + 2) * npx + cx * 2 + 1
            m5 = (cy * 2 + 1) * npx + cx * 2 + 2
            connect[elem_idx] = [br, tl, bl, m1, m2, m3]
            elem_idx += 1
            connect[elem_idx] = [tr, tl, br, m4, m2, m5]
            elem_idx += 1
    return (coords, connect)