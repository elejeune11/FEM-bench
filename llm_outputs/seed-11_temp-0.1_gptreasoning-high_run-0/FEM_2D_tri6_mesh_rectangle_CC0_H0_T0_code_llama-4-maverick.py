def FEM_2D_tri6_mesh_rectangle_CC0_H0_T0(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('xl < xh and yl < yh must hold')
    if nx < 1 or ny < 1:
        raise ValueError('nx >= 1 and ny >= 1 must hold')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    Nnodes = npx * npy
    Ne = 2 * nx * ny
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    (ix, iy) = np.meshgrid(np.arange(npx), np.arange(npy), indexing='xy')
    coords = np.column_stack((xl + 0.5 * dx * ix.flatten(), yl + 0.5 * dy * iy.flatten()))
    connect = np.zeros((Ne, 6), dtype=np.int64)
    elem_id = 0
    for cy in range(ny):
        for cx in range(nx):
            bottom_left = cy * 2 * npx + cx * 2
            bottom_right = bottom_left + 2
            top_left = bottom_left + 2 * npx
            top_right = top_left + 2
            connect[elem_id, :] = [bottom_right, top_left, bottom_left, top_left - 1, top_left + npx, bottom_left + 1]
            elem_id += 1
            connect[elem_id, :] = [top_right, top_left, bottom_right, top_left + 1, bottom_right + npx, bottom_right - 1]
            elem_id += 1
    return (coords, connect)