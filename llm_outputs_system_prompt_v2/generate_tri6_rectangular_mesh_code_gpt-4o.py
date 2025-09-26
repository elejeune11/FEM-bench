def generate_tri6_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if xl >= xh or yl >= yh:
        raise ValueError('Invalid domain bounds: xl must be < xh and yl must be < yh.')
    if nx < 1 or ny < 1:
        raise ValueError('Invalid subdivisions: nx and ny must be >= 1.')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    x_coords = np.linspace(xl, xh, npx)
    y_coords = np.linspace(yl, yh, npy)
    (xv, yv) = np.meshgrid(x_coords, y_coords, indexing='xy')
    coords = np.vstack((xv.flatten(), yv.flatten())).T
    connect = np.zeros((2 * nx * ny, 6), dtype=np.int64)
    element_index = 0
    for cy in range(ny):
        for cx in range(nx):
            ix = 2 * cx
            iy = 2 * cy
            n1 = (iy + 1) * npx + (ix + 2)
            n2 = (iy + 2) * npx + ix
            n3 = iy * npx + ix
            n4 = (iy + 1) * npx + (ix + 1)
            n5 = (iy + 1) * npx + ix
            n6 = iy * npx + (ix + 1)
            connect[element_index] = [n1, n2, n3, n4, n5, n6]
            element_index += 1
            n1 = (iy + 2) * npx + (ix + 2)
            n2 = (iy + 2) * npx + ix
            n3 = (iy + 1) * npx + (ix + 2)
            n4 = (iy + 2) * npx + (ix + 1)
            n5 = (iy + 1) * npx + (ix + 1)
            n6 = (iy + 1) * npx + (ix + 2)
            connect[element_index] = [n1, n2, n3, n4, n5, n6]
            element_index += 1
    return (coords, connect)