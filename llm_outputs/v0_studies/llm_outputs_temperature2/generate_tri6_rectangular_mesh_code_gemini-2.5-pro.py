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
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh.')
    if nx < 1 or ny < 1:
        raise ValueError('Number of subdivisions nx and ny must be at least 1.')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    x_pts = np.linspace(xl, xh, npx, dtype=np.float64)
    y_pts = np.linspace(yl, yh, npy, dtype=np.float64)
    (xx, yy) = np.meshgrid(x_pts, y_pts, indexing='xy')
    coords = np.stack([xx.flatten('C'), yy.flatten('C')], axis=1)
    num_elements = 2 * nx * ny
    connect = np.empty((num_elements, 6), dtype=np.int64)
    (cx_all, cy_all) = np.meshgrid(np.arange(nx), np.arange(ny))
    cx = cx_all.flatten('C')
    cy = cy_all.flatten('C')
    ix_bl = 2 * cx
    iy_bl = 2 * cy
    id_bl = iy_bl * npx + ix_bl
    id_br = iy_bl * npx + (ix_bl + 2)
    id_tl = (iy_bl + 2) * npx + ix_bl
    id_tr = (iy_bl + 2) * npx + (ix_bl + 2)
    id_mb = iy_bl * npx + (ix_bl + 1)
    id_mr = (iy_bl + 1) * npx + (ix_bl + 2)
    id_mt = (iy_bl + 2) * npx + (ix_bl + 1)
    id_ml = (iy_bl + 1) * npx + ix_bl
    id_c = (iy_bl + 1) * npx + (ix_bl + 1)
    connect[0::2] = np.stack([id_br, id_tl, id_bl, id_c, id_ml, id_mb], axis=1)
    connect[1::2] = np.stack([id_tr, id_tl, id_br, id_mt, id_c, id_mr], axis=1)
    return (coords, connect)