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
    coords = np.column_stack((X.ravel(), Y.ravel()))
    connect = []
    for cy in range(ny):
        for cx in range(nx):
            base_row = 2 * cy
            base_col = 2 * cx
            n00 = base_row * npx + base_col
            n10 = n00 + 1
            n20 = n00 + 2
            n01 = n00 + npx
            n11 = n01 + 1
            n21 = n01 + 2
            n02 = n00 + 2 * npx
            n12 = n02 + 1
            n22 = n02 + 2
            tri1 = [n20, n02, n00, n11, n01, n10]
            tri2 = [n22, n02, n20, n12, n11, n21]
            connect.extend([tri1, tri2])
    return (coords, np.array(connect, dtype=np.int64))