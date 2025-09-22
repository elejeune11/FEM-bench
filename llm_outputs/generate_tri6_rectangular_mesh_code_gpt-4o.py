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
            bl = 2 * cy * npx + 2 * cx
            br = bl + 2
            tl = bl + 2 * npx
            tr = tl + 2
            mbl_tr = bl + 1
            mtl_br = bl + npx + 1
            connect[element_index] = [br, tl, bl, mtl_br, mbl_tr, br + npx]
            element_index += 1
            connect[element_index] = [tr, tl, br, tr - 1, mtl_br, tr - npx]
            element_index += 1
    return (coords, connect)