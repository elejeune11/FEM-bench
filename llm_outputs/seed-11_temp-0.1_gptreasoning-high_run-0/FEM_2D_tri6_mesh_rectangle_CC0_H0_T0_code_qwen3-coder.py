def FEM_2D_tri6_mesh_rectangle_CC0_H0_T0(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('xl < xh and yl < yh required')
    if nx < 1 or ny < 1:
        raise ValueError('nx ≥ 1 and ny ≥ 1 required')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    ix = np.arange(npx)
    iy = np.arange(npy)
    (IX, IY) = np.meshgrid(ix, iy, indexing='xy')
    X = xl + 0.5 * dx * IX
    Y = yl + 0.5 * dy * IY
    coords = np.column_stack((X.ravel(), Y.ravel())).astype(np.float64)

    def node_id(ix, iy):
        return iy * npx + ix
    connect_list = []
    for cy in range(ny):
        for cx in range(nx):
            bl = node_id(2 * cx, 2 * cy)
            br = node_id(2 * cx + 2, 2 * cy)
            tl = node_id(2 * cx, 2 * cy + 2)
            tr = node_id(2 * cx + 2, 2 * cy + 2)
            b = node_id(2 * cx + 1, 2 * cy)
            r = node_id(2 * cx + 2, 2 * cy + 1)
            t = node_id(2 * cx + 1, 2 * cy + 2)
            l = node_id(2 * cx, 2 * cy + 1)
            c = node_id(2 * cx + 1, 2 * cy + 1)
            connect_list.append([br, tl, bl, c, l, b])
            connect_list.append([tr, tl, br, t, c, r])
    connect = np.array(connect_list, dtype=np.int64)
    return (coords, connect)