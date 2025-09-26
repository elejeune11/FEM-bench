def generate_tri6_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be >= 1')
    if xl >= xh or yl >= yh:
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x = np.linspace(xl, xh, npx)
    y = np.linspace(yl, yh, npy)
    (X, Y) = np.meshgrid(x, y, indexing='xy')
    coords = np.column_stack((X.ravel(), Y.ravel()))
    n_elements = 2 * nx * ny
    connect = np.zeros((n_elements, 6), dtype=np.int64)

    def node_id(ix, iy):
        return iy * npx + ix
    elem = 0
    for cy in range(ny):
        for cx in range(nx):
            i0 = 2 * cx
            j0 = 2 * cy
            bl = node_id(i0, j0)
            br = node_id(i0 + 2, j0)
            tl = node_id(i0, j0 + 2)
            tr = node_id(i0 + 2, j0 + 2)
            bm = node_id(i0 + 1, j0)
            rm = node_id(i0 + 2, j0 + 1)
            tm = node_id(i0 + 1, j0 + 2)
            lm = node_id(i0, j0 + 1)
            cm = node_id(i0 + 1, j0 + 1)
            connect[elem] = [br, tl, bl, cm, lm, bm]
            elem += 1
            connect[elem] = [tr, tl, br, tm, cm, rm]
            elem += 1
    return (coords, connect)