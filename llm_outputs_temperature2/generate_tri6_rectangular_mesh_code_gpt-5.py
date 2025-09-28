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
    import numpy as np
    if not xl < xh or not yl < yh:
        raise ValueError('Invalid domain bounds: require xl < xh and yl < yh.')
    if nx < 1 or ny < 1:
        raise ValueError('Invalid subdivisions: require nx >= 1 and ny >= 1.')
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    xs = xl + 0.5 * dx * np.arange(npx, dtype=np.float64)
    ys = yl + 0.5 * dy * np.arange(npy, dtype=np.float64)
    (X, Y) = np.meshgrid(xs, ys, indexing='xy')
    coords = np.column_stack((X.ravel(order='C'), Y.ravel(order='C'))).astype(np.float64, copy=False)

    def nid(ix: int, iy: int) -> int:
        return iy * npx + ix
    ne = 2 * nx * ny
    connect = np.empty((ne, 6), dtype=np.int64)
    e = 0
    for cy in range(ny):
        iy0 = 2 * cy
        iy1 = iy0 + 1
        iy2 = iy0 + 2
        for cx in range(nx):
            ix0 = 2 * cx
            ix1 = ix0 + 1
            ix2 = ix0 + 2
            bl = nid(ix0, iy0)
            br = nid(ix2, iy0)
            tl = nid(ix0, iy2)
            tr = nid(ix2, iy2)
            mid_bottom = nid(ix1, iy0)
            mid_top = nid(ix1, iy2)
            mid_left = nid(ix0, iy1)
            mid_right = nid(ix2, iy1)
            mid_diag = nid(ix1, iy1)
            connect[e, 0] = br
            connect[e, 1] = tl
            connect[e, 2] = bl
            connect[e, 3] = mid_diag
            connect[e, 4] = mid_left
            connect[e, 5] = mid_bottom
            e += 1
            connect[e, 0] = tr
            connect[e, 1] = tl
            connect[e, 2] = br
            connect[e, 3] = mid_top
            connect[e, 4] = mid_diag
            connect[e, 5] = mid_right
            e += 1
    return (coords, connect)