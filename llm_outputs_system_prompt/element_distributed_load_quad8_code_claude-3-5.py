def element_distributed_load_quad8(face: int, node_coords: np.ndarray, traction: np.ndarray, num_gauss_pts: int) -> np.ndarray:
    gauss_pts = {1: np.array([0.0]), 2: np.array([-1 / np.sqrt(3), 1 / np.sqrt(3)]), 3: np.array([-np.sqrt(0.6), 0.0, np.sqrt(0.6)])}
    gauss_wts = {1: np.array([2.0]), 2: np.array([1.0, 1.0]), 3: np.array([5 / 9, 8 / 9, 5 / 9])}
    edge_map = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    r_elem = np.zeros(16)
    edge_nodes = edge_map[face]

    def shape_funcs(s):
        N = np.array([-0.5 * s * (1 - s), 1 - s ** 2, 0.5 * s * (1 + s)])
        dN = np.array([-0.5 + s, -2 * s, 0.5 + s])
        return (N, dN)
    s_pts = gauss_pts[num_gauss_pts]
    w_pts = gauss_wts[num_gauss_pts]
    for (i, s) in enumerate(s_pts):
        (N, dN) = shape_funcs(s)
        edge_coords = node_coords[edge_nodes]
        dx_ds = np.dot(dN, edge_coords)
        J = np.sqrt(np.sum(dx_ds ** 2))
        for (j, node) in enumerate(edge_nodes):
            r_elem[2 * node:2 * node + 2] += N[j] * traction * J * w_pts[i]
    return r_elem