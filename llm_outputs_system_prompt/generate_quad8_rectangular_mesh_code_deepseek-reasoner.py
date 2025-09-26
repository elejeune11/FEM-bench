def generate_quad8_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh')
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be at least 1')
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    total_refined_nodes = npx * npy
    center_nodes = nx * ny
    Nnodes = total_refined_nodes - center_nodes
    coords = np.zeros((Nnodes, 2), dtype=np.float64)
    node_id_map = np.full((npx, npy), -1, dtype=np.int64)
    node_id = 0
    for iy in range(npy):
        for ix in range(npx):
            if not (ix % 2 == 1 and iy % 2 == 1):
                x = xl + 0.5 * dx * ix
                y = yl + 0.5 * dy * iy
                coords[node_id] = [x, y]
                node_id_map[ix, iy] = node_id
                node_id += 1
    Ne = nx * ny
    connect = np.zeros((Ne, 8), dtype=np.int64)
    for cy in range(ny):
        for cx in range(nx):
            elem_id = cy * nx + cx
            ix0 = 2 * cx
            iy0 = 2 * cy
            n1 = node_id_map[ix0, iy0]
            n2 = node_id_map[ix0 + 2, iy0]
            n3 = node_id_map[ix0 + 2, iy0 + 2]
            n4 = node_id_map[ix0, iy0 + 2]
            n5 = node_id_map[ix0 + 1, iy0]
            n6 = node_id_map[ix0 + 2, iy0 + 1]
            n7 = node_id_map[ix0 + 1, iy0 + 2]
            n8 = node_id_map[ix0, iy0 + 1]
            connect[elem_id] = [n1, n2, n3, n4, n5, n6, n7, n8]
    return (coords, connect)