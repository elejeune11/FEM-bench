def generate_quad8_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh')
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be at least 1')
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    x_refined = np.linspace(xl, xh, npx, dtype=np.float64)
    y_refined = np.linspace(yl, yh, npy, dtype=np.float64)
    node_id_map = np.full((npy, npx), -1, dtype=np.int64)
    node_count = 0
    coords_list = []
    for iy in range(npy):
        for ix in range(npx):
            if not (ix % 2 == 1 and iy % 2 == 1):
                node_id_map[iy, ix] = node_count
                coords_list.append([x_refined[ix], y_refined[iy]])
                node_count += 1
    coords = np.array(coords_list, dtype=np.float64)
    connect = []
    for cy in range(ny):
        for cx in range(nx):
            ix0 = 2 * cx
            iy0 = 2 * cy
            n1 = node_id_map[iy0, ix0]
            n2 = node_id_map[iy0, ix0 + 2]
            n3 = node_id_map[iy0 + 2, ix0 + 2]
            n4 = node_id_map[iy0 + 2, ix0]
            n5 = node_id_map[iy0, ix0 + 1]
            n6 = node_id_map[iy0 + 1, ix0 + 2]
            n7 = node_id_map[iy0 + 2, ix0 + 1]
            n8 = node_id_map[iy0 + 1, ix0]
            connect.append([n1, n2, n3, n4, n5, n6, n7, n8])
    return (coords, np.array(connect, dtype=np.int64))