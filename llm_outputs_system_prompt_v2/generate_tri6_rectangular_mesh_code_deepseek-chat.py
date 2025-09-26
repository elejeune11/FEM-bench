def generate_tri6_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh')
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be >= 1')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x_coords = np.linspace(xl, xh, npx, dtype=np.float64)
    y_coords = np.linspace(yl, yh, npy, dtype=np.float64)
    (X, Y) = np.meshgrid(x_coords, y_coords, indexing='xy')
    coords = np.column_stack((X.flatten(), Y.flatten()))
    n_elements = 2 * nx * ny
    connect = np.zeros((n_elements, 6), dtype=np.int64)
    element_index = 0
    for cy in range(ny):
        for cx in range(nx):
            base_x = 2 * cx
            base_y = 2 * cy
            bottom_left = base_y * npx + base_x
            bottom_right = base_y * npx + base_x + 2
            top_left = (base_y + 2) * npx + base_x
            top_right = (base_y + 2) * npx + base_x + 2
            mid_bottom = base_y * npx + base_x + 1
            mid_right = (base_y + 1) * npx + base_x + 2
            mid_top = (base_y + 2) * npx + base_x + 1
            mid_left = (base_y + 1) * npx + base_x
            center = (base_y + 1) * npx + base_x + 1
            connect[element_index] = [bottom_right, top_left, bottom_left, center, mid_left, mid_bottom]
            element_index += 1
            connect[element_index] = [top_right, top_left, bottom_right, mid_top, center, mid_right]
            element_index += 1
    return (coords, connect)