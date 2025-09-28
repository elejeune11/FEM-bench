def generate_quad8_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('Invalid domain bounds: xl must be < xh and yl must be < yh.')
    if nx < 1 or ny < 1:
        raise ValueError('Invalid subdivisions: nx and ny must be >= 1.')
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    coords = []
    node_id_map = {}
    id_counter = 0
    for iy in range(npy):
        for ix in range(npx):
            if not (ix % 2 == 1 and iy % 2 == 1):
                x = xl + 0.5 * dx * ix
                y = yl + 0.5 * dy * iy
                coords.append([x, y])
                node_id_map[ix, iy] = id_counter
                id_counter += 1
    coords = np.array(coords, dtype=np.float64)
    connect = []
    for cy in range(ny):
        for cx in range(nx):
            (ix0, iy0) = (2 * cx, 2 * cy)
            N1 = node_id_map[ix0, iy0]
            N2 = node_id_map[ix0 + 2, iy0]
            N3 = node_id_map[ix0 + 2, iy0 + 2]
            N4 = node_id_map[ix0, iy0 + 2]
            N5 = node_id_map[ix0 + 1, iy0]
            N6 = node_id_map[ix0 + 2, iy0 + 1]
            N7 = node_id_map[ix0 + 1, iy0 + 2]
            N8 = node_id_map[ix0, iy0 + 1]
            connect.append([N1, N2, N3, N4, N5, N6, N7, N8])
    connect = np.array(connect, dtype=np.int64)
    return (coords, connect)