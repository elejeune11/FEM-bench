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
    x_coords = np.linspace(xl, xh, npx, dtype=np.float64)
    y_coords = np.linspace(yl, yh, npy, dtype=np.float64)
    (xx, yy) = np.meshgrid(x_coords, y_coords, indexing='xy')
    coords = np.vstack((xx.flatten(), yy.flatten())).T
    num_cells = nx * ny
    cx_all = np.tile(np.arange(nx), ny)
    cy_all = np.repeat(np.arange(ny), nx)
    ix_base_all = 2 * cx_all
    iy_base_all = 2 * cy_all
    id_BL_all = iy_base_all * npx + ix_base_all
    id_BM_all = iy_base_all * npx + (ix_base_all + 1)
    id_BR_all = iy_base_all * npx + (ix_base_all + 2)
    id_ML_all = (iy_base_all + 1) * npx + ix_base_all
    id_C_all = (iy_base_all + 1) * npx + (ix_base_all + 1)
    id_MR_all = (iy_base_all + 1) * npx + (ix_base_all + 2)
    id_TL_all = (iy_base_all + 2) * npx + ix_base_all
    id_TM_all = (iy_base_all + 2) * npx + (ix_base_all + 1)
    id_TR_all = (iy_base_all + 2) * npx + (ix_base_all + 2)
    connect_tri1 = np.vstack([id_BR_all, id_TL_all, id_BL_all, id_C_all, id_ML_all, id_BM_all]).T
    connect_tri2 = np.vstack([id_TR_all, id_TL_all, id_BR_all, id_TM_all, id_C_all, id_MR_all]).T
    num_elements = 2 * num_cells
    connect = np.zeros((num_elements, 6), dtype=np.int64)
    connect[0::2, :] = connect_tri1
    connect[1::2, :] = connect_tri2
    return (coords, connect)