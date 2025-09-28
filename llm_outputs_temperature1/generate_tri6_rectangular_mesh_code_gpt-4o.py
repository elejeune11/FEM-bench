def generate_tri6_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('xl must be less than xh and yl must be less than yh.')
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be at least 1.')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    coords = np.zeros((npx * npy, 2), dtype=np.float64)
    for iy in range(npy):
        for ix in range(npx):
            node_id = iy * npx + ix
            coords[node_id] = [xl + 0.5 * dx * ix, yl + 0.5 * dy * iy]
    connect = np.zeros((2 * nx * ny, 6), dtype=np.int64)
    element_id = 0
    for cy in range(ny):
        for cx in range(nx):
            br = (2 * cy + 0) * npx + (2 * cx + 2)
            tl = (2 * cy + 2) * npx + (2 * cx + 0)
            bl = (2 * cy + 0) * npx + (2 * cx + 0)
            tr = (2 * cy + 2) * npx + (2 * cx + 2)
            m1 = (2 * cy + 1) * npx + (2 * cx + 1)
            m2 = (2 * cy + 0) * npx + (2 * cx + 1)
            m3 = (2 * cy + 1) * npx + (2 * cx + 0)
            m4 = (2 * cy + 2) * npx + (2 * cx + 1)
            m5 = (2 * cy + 1) * npx + (2 * cx + 2)
            connect[element_id] = [br, tl, bl, m1, m3, m2]
            element_id += 1
            connect[element_id] = [tr, tl, br, m4, m1, m5]
            element_id += 1
    return (coords, connect)