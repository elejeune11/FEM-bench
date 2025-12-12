def FEM_2D_tri6_mesh_rectangle_CC0_H0_T0(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be greater than or equal to 1')
    if xl >= xh or yl >= yh:
        raise ValueError('xl < xh and yl < yh must be satisfied')
    (npx, npy) = (2 * nx + 1, 2 * ny + 1)
    Nnodes = npx * npy
    Ne = 2 * nx * ny
    coords = np.empty((Nnodes, 2), dtype=np.float64)
    connect = np.empty((Ne, 6), dtype=np.int64)
    (ix, iy) = np.meshgrid(np.arange(npx), np.arange(npy), indexing='xy')
    ix = ix.flatten()
    iy = iy.flatten()
    coords[:, 0] = xl + (xh - xl) * ix / (2 * nx)
    coords[:, 1] = yl + (yh - yl) * iy / (2 * ny)
    for cy in range(ny):
        for cx in range(nx):
            cell_offset = cy * npx + cx
            N1 = cell_offset
            N2 = cell_offset + npx + 1
            N3 = cell_offset + 1
            N4 = cell_offset + npx
            N5 = cell_offset + npx + 1 + 1
            N6 = cell_offset + 1 + npx
            connect[2 * (cx + cy * nx), :] = [N3, N2, N1, N6, N5, N4]
            connect[2 * (cx + cy * nx) + 1, :] = [N2, N1, N3, N5, N4, N6]
    return (coords, connect)