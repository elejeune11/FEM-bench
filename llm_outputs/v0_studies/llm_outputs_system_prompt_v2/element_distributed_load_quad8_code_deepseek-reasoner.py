def element_distributed_load_quad8(face: int, node_coords: np.ndarray, traction: np.ndarray, num_gauss_pts: int) -> np.ndarray:
    face_nodes = {0: [0, 4, 1], 1: [1, 5, 2], 2: [2, 6, 3], 3: [3, 7, 0]}
    gauss_points = {1: (np.array([0.0]), np.array([2.0])), 2: (np.array([-0.5773502691896257, 0.5773502691896257]), np.array([1.0, 1.0])), 3: (np.array([-0.7745966692414834, 0.0, 0.7745966692414834]), np.array([0.5555555555555556, 0.8888888888888888, 0.5555555555555556]))}
    nodes = face_nodes[face]
    edge_coords = node_coords[nodes]
    (xi, weights) = gauss_points[num_gauss_pts]
    r_edge = np.zeros(6)
    for i in range(len(xi)):
        s = xi[i]
        w = weights[i]
        N = np.array([s * (s - 1) / 2, 1 - s ** 2, s * (s + 1) / 2])
        dNds = np.array([s - 0.5, -2 * s, s + 0.5])
        dxds = dNds @ edge_coords[:, 0]
        dyds = dNds @ edge_coords[:, 1]
        jac = np.sqrt(dxds ** 2 + dyds ** 2)
        r_edge[0:2] += N[0] * traction * jac * w
        r_edge[2:4] += N[1] * traction * jac * w
        r_edge[4:6] += N[2] * traction * jac * w
    r_elem = np.zeros(16)
    r_elem[2 * nodes[0]:2 * nodes[0] + 2] = r_edge[0:2]
    r_elem[2 * nodes[1]:2 * nodes[1] + 2] = r_edge[2:4]
    r_elem[2 * nodes[2]:2 * nodes[2] + 2] = r_edge[4:6]
    return r_elem