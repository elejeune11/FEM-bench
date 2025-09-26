def generate_tri6_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh')
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be ≥ 1')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x_coords = np.linspace(xl, xh, npx, dtype=np.float64)
    y_coords = np.linspace(yl, yh, npy, dtype=np.float64)
    (xx, yy) = np.meshgrid(x_coords, y_coords, indexing='xy')
    coords = np.column_stack((xx.ravel(), yy.ravel()))
    n_elements = 2 * nx * ny
    connect = np.zeros((n_elements, 6), dtype=np.int64)
    element_idx = 0
    for cy in range(ny):
        for cx in range(nx):
            base_x = 2 * cx
            base_y = 2 * cy
            bottom_left = base_y * npx + base_x
            bottom_right = bottom_left + 2
            top_left = (base_y + 2) * npx + base_x
            top_right = top_left + 2
            mid_bottom = bottom_left + 1
            mid_left = (base_y + 1) * npx + base_x
            mid_center = (base_y + 1) * npx + base_x + 1
            mid_top = top_left + 1
            mid_right = (base_y + 1) * npx + base_x + 2
            connect[element_idx] = [bottom_right, top_left, bottom_left, mid_center, mid_left, mid_bottom]
            connect[element_idx + 1] = [top_right, top_left, bottom_right, mid_top, mid_center, mid_right]
            element_idx += 2
    return (coords, connect)