def generate_quad8_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh')
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be ≥ 1')
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    node_id = 0
    node_map = np.full((npy, npx), -1, dtype=np.int64)
    coords_list = []
    for iy in range(npy):
        for ix in range(npx):
            if not (ix % 2 == 1 and iy % 2 == 1):
                node_map[iy, ix] = node_id
                coords_list.append([xl + 0.5 * dx * ix, yl + 0.5 * dy * iy])
                node_id += 1
    coords = np.array(coords_list, dtype=np.float64)
    connect = []
    for cy in range(ny):
        for cx in range(nx):
            (ix0, iy0) = (2 * cx, 2 * cy)
            element_nodes = [node_map[iy0, ix0], node_map[iy0, ix0 + 2], node_map[iy0 + 2, ix0 + 2], node_map[iy0 + 2, ix0], node_map[iy0, ix0 + 1], node_map[iy0 + 1, ix0 + 2], node_map[iy0 + 2, ix0 + 1], node_map[iy0 + 1, ix0]]
            connect.append(element_nodes)
    return (coords, np.array(connect, dtype=np.int64))