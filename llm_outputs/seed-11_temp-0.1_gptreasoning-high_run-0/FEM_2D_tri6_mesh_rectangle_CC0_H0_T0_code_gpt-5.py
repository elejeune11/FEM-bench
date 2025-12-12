def FEM_2D_tri6_mesh_rectangle_CC0_H0_T0(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
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
        Node coordinates as specified above, with
        Nnodes = (2*nx + 1) * (2*ny + 1).
    connect : (Ne, 6) int64 ndarray
        Tri6 connectivity using the exact ordering defined above, with
        Ne = 2 * nx * ny.
    Raises
    ------
    ValueError
        If nx < 1 or ny < 1, or if xl >= xh or yl >= yh.
    Notes
    -----
    • The ordering of nodes and emitted elements is deterministic and part of the API.
    • Corner nodes are consistently oriented CCW for each triangle.
    • Midside nodes lie exactly at the arithmetic mean of their adjacent corners and
      coincide with half-step grid points; they are shared (no duplication).
    • The mesh is conforming: shared edges reference identical global node IDs.
    """
    if not xl < xh or not yl < yh:
        raise ValueError('Invalid bounds: require xl < xh and yl < yh.')
    if nx < 1 or ny < 1:
        raise ValueError('Invalid subdivisions: require nx >= 1 and ny >= 1.')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    ix = np.arange(npx, dtype=np.int64)
    iy = np.arange(npy, dtype=np.int64)
    (Xg, Yg) = np.meshgrid(ix, iy, indexing='xy')
    x_coords = xl + 0.5 * dx * Xg
    y_coords = yl + 0.5 * dy * Yg
    coords = np.stack((x_coords.ravel(order='C'), y_coords.ravel(order='C')), axis=1).astype(np.float64, copy=False)
    Ne = 2 * nx * ny
    connect = np.empty((Ne, 6), dtype=np.int64)
    e = 0
    for cy in range(ny):
        iyB = 2 * cy
        iyT = iyB + 2
        for cx in range(nx):
            ixL = 2 * cx
            ixR = ixL + 2
            bl = iyB * npx + ixL
            br = iyB * npx + ixR
            tl = iyT * npx + ixL
            tr = iyT * npx + ixR
            mb = iyB * npx + (ixL + 1)
            ml = (iyB + 1) * npx + ixL
            mr = (iyB + 1) * npx + ixR
            mt = iyT * npx + (ixL + 1)
            md = (iyB + 1) * npx + (ixL + 1)
            connect[e, 0] = br
            connect[e, 1] = tl
            connect[e, 2] = bl
            connect[e, 3] = md
            connect[e, 4] = ml
            connect[e, 5] = mb
            e += 1
            connect[e, 0] = tr
            connect[e, 1] = tl
            connect[e, 2] = br
            connect[e, 3] = mt
            connect[e, 4] = md
            connect[e, 5] = mr
            e += 1
    return (coords, connect)