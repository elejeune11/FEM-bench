def generate_tri6_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('Invalid domain bounds: xl must be < xh and yl must be < yh.')
    if nx < 1 or ny < 1:
        raise ValueError('Invalid subdivisions: nx and ny must be >= 1.')
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
            n0 = 2 * cy * npx + (2 * cx + 2)
            n1 = (2 * cy + 2) * npx + 2 * cx
            n2 = 2 * cy * npx + 2 * cx
            n3 = (2 * cy + 2) * npx + (2 * cx + 2)
            n4 = (2 * cy + 2) * npx + (2 * cx + 1)
            n5 = (2 * cy + 1) * npx + 2 * cx
            n6 = (2 * cy + 1) * npx + (2 * cx + 2)
            n7 = 2 * cy * npx + (2 * cx + 1)
            connect[element_id] = [n0, n1, n2, n4, n5, n7]
            element_id += 1
            connect[element_id] = [n3, n1, n0, n4, n6, n7]
            element_id += 1
    return (coords, connect)