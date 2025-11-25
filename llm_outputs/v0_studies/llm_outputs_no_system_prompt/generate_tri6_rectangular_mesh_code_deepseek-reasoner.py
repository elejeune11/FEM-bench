def generate_tri6_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh')
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be at least 1')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x_coords = xl + 0.5 * dx * np.arange(npx, dtype=np.float64)
    y_coords = yl + 0.5 * dy * np.arange(npy, dtype=np.float64)
    (X, Y) = np.meshgrid(x_coords, y_coords, indexing='xy')
    coords = np.column_stack((X.ravel(), Y.ravel()))
    n_elements = 2 * nx * ny
    connect = np.zeros((n_elements, 6), dtype=np.int64)
    element_idx = 0
    for cy in range(ny):
        for cx in range(nx):
            base_x = 2 * cx
            base_y = 2 * cy
            n1 = base_y * npx + (base_x + 2)
            n2 = (base_y + 2) * npx + base_x
            n3 = base_y * npx + base_x
            n4 = (base_y + 1) * npx + (base_x + 1)
            n5 = (base_y + 1) * npx + base_x
            n6 = base_y * npx + (base_x + 1)
            connect[element_idx] = [n1, n2, n3, n4, n5, n6]
            element_idx += 1
            n1 = (base_y + 2) * npx + (base_x + 2)
            n2 = (base_y + 2) * npx + base_x
            n3 = base_y * npx + (base_x + 2)
            n4 = (base_y + 2) * npx + (base_x + 1)
            n5 = (base_y + 1) * npx + (base_x + 1)
            n6 = (base_y + 1) * npx + (base_x + 2)
            connect[element_idx] = [n1, n2, n3, n4, n5, n6]
            element_idx += 1
    return (coords, connect)