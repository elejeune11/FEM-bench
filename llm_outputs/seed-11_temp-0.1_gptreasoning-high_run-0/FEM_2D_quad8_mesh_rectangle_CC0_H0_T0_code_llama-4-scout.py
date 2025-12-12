def FEM_2D_quad8_mesh_rectangle_CC0_H0_T0(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be greater than or equal to 1')
    if xl >= xh or yl >= yh:
        raise ValueError('xl < xh and yl < yh must be satisfied')
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    ix = np.arange(npx)
    iy = np.arange(npy)
    x = xl + 0.5 * dx * ix
    y = yl + 0.5 * dy * iy
    (x, y) = np.meshgrid(x, y, indexing='xy')
    x = x.flatten()
    y = y.flatten()
    mask = ~(np.mod(ix, 2) == 1 & np.mod(iy, 2) == 1)
    mask = mask.flatten()
    coords = np.column_stack((x[mask], y[mask]))
    Ne = nx * ny
    connect = np.zeros((Ne, 8), dtype=np.int64)
    for cy in range(ny):
        for cx in range(nx):
            ix0 = 2 * cx
            iy0 = 2 * cy
            N1 = iy0 * npx + ix0
            N2 = iy0 * npx + ix0 + 2
            N3 = (iy0 + 2) * npx + ix0 + 2
            N4 = (iy0 + 2) * npx + ix0
            N5 = iy0 * npx + ix0 + 1
            N6 = (iy0 + 1) * npx + ix0 + 2
            N7 = (iy0 + 2) * npx + ix0 + 1
            N8 = (iy0 + 1) * npx + ix0
            connect[cy * nx + cx] = [N1, N2, N3, N4, N5, N6, N7, N8]
    return (coords, connect)