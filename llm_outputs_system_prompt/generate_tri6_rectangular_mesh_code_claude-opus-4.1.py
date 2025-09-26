def generate_tri6_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a quadratic triangular (Tri6 / 6-node triangle) mesh on a rectangular domain.
    The domain [xl, xh] × [yl, yh] is subdivided into `nx × ny` rectangular cells.
    Each rectangle is split into two 6-node triangular elements, with corner nodes
    ordered counter-clockwise and midside nodes placed at edge midpoints.
    Reproducibility contract (for identical outputs across implementations)
    ----------------------------------------------------------------------
    Preconditions
    • xl < xh and yl < yh; else raise ValueError.
    • nx ≥ 1 and ny ≥ 1; else raise ValueError.
    Grid, node IDs, and coordinates
    • npx = 2*nx + 1,  npy = 2*ny + 1.
    • Global node IDs are zero-based and assigned in row-major order with x varying fastest:
        node_id(ix, iy) = iy * npx + ix,   where 0 ≤ ix < npx and 0 ≤ iy < npy.
      Equivalently: build with meshgrid(indexing="xy") and flatten in C-order (row-major).
    • Let dx = (xh - xl)/nx and dy = (yh - yl)/ny. Then coordinates are
        coords[node_id(ix, iy)] = [ xl + 0.5*dx*ix ,  yl + 0.5*dy*iy ]
      computed in float64 (no averaging from other nodes).
    Cell traversal and element emission
    • Traverse cells row-major: cy = 0..ny-1 (bottom→top), for each cy, cx = 0..nx-1 (left→right).
    • Each cell is split along the diagonal from the bottom-right corner to the top-left corner.
    • Emit exactly two Tri6 elements per cell in this order:
        1) First triangle (corners CCW): N1 = bottom-right, N2 = top-left, N3 = bottom-left.
           Midsides: N4 on (N1,N2), N5 on (N2,N3), N6 on (N3,N1).
        2) Second triangle (corners CCW): N1 = top-right,   N2 = top-left, N3 = bottom-right.
           Midsides: N4 on (N1,N2), N5 on (N2,N3), N6 on (N3,N1).
      Do not reorder elements after emission. Midside nodes must reference the shared grid nodes.
    Types and shapes
    • coords is a ( (2*nx+1)*(2*ny+1), 2 ) ndarray with dtype float64.
    • connect is a ( 2*nx*ny, 6 ) ndarray with dtype int64.
      Each row is [N1, N2, N3, N4, N5, N6] as specified above.
    Parameters
    ----------
    xl, yl, xh, yh : float
        Domain bounds with xl < xh and yl < yh.
    nx, ny : int
        Number of rectangular subdivisions in x and y (each ≥ 1).
    Returns
    -------
    coords : (Nnodes, 2) float64 ndarray
        Node coordinates as specified in the contract above.
    connect : (Ne, 6) int64 ndarray
        Tri6 connectivity using the exact ordering defined above.
    Raises
    ------
    ValueError
        If nx < 1 or ny < 1, or if xl >= xh or yl >= yh.
    Notes
    -----
    • Corner nodes are consistently oriented CCW for each triangle.
    • Midside nodes lie exactly at the arithmetic mean of their adjacent corners and
      coincide with half-step grid points; they are shared (no duplication).
    • The mesh is conforming: shared edges reference identical global node IDs.
    """
    if xl >= xh or yl >= yh:
        raise ValueError
    if nx < 1 or ny < 1:
        raise ValueError
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    coords = np.zeros((npx * npy, 2), dtype=np.float64)
    for iy in range(npy):
        for ix in range(npx):
            node_id = iy * npx + ix
            coords[node_id, 0] = xl + 0.5 * dx * ix
            coords[node_id, 1] = yl + 0.5 * dy * iy
    connect = np.zeros((2 * nx * ny, 6), dtype=np.int64)
    elem_idx = 0
    for cy in range(ny):
        for cx in range(nx):
            ix_bl = 2 * cx
            iy_bl = 2 * cy
            ix_br = 2 * cx + 2
            iy_br = 2 * cy
            ix_tl = 2 * cx
            iy_tl = 2 * cy + 2
            ix_tr = 2 * cx + 2
            iy_tr = 2 * cy + 2
            bl = iy_bl * npx + ix_bl
            br = iy_br * npx + ix_br
            tl = iy_tl * npx + ix_tl
            tr = iy_tr * npx + ix_tr
            ix_b_mid = 2 * cx + 1
            iy_b_mid = 2 * cy
            ix_t_mid = 2 * cx + 1
            iy_t_mid = 2 * cy + 2
            ix_l_mid = 2 * cx
            iy_l_mid = 2 * cy + 1
            ix_r_mid = 2 * cx + 2
            iy_r_mid = 2 * cy + 1
            ix_c_mid = 2 * cx + 1
            iy_c_mid = 2 * cy + 1
            b_mid = iy_b_mid * npx + ix_b_mid
            t_mid = iy_t_mid * npx + ix_t_mid
            l_mid = iy_l_mid * npx + ix_l_mid
            r_mid = iy_r_mid * npx + ix_r_mid
            c_mid = iy_c_mid * npx + ix_c_mid
            connect[elem_idx, 0] = br
            connect[elem_idx, 1] = tl
            connect[elem_idx, 2] = bl
            connect[elem_idx, 3] = c_mid
            connect[elem_idx, 4] = l_mid
            connect[elem_idx, 5] = b_mid
            elem_idx += 1
            connect[elem_idx, 0] = tr
            connect[elem_idx, 1] = tl
            connect[elem_idx, 2] = br
            connect[elem_idx, 3] = t_mid
            connect[elem_idx, 4] = c_mid
            connect[elem_idx, 5] = r_mid
            elem_idx += 1
    return (coords, connect)