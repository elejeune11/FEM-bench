def generate_quad8_rectangular_mesh(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Generate a 2D mesh of 8-node quadrilateral (Quad8) elements on a rectangular domain.
    The domain [xl, xh] × [yl, yh] is subdivided into `nx × ny` rectangular cells.
    Each cell is represented by a quadratic quadrilateral with 4 corner nodes and
    4 midside nodes (no interior/center node). Node coordinates are taken from a
    refined half-step grid to guarantee conformity between adjacent elements.
    Reproducibility contract (for identical outputs across implementations)
    ----------------------------------------------------------------------
    Preconditions
    • xl < xh and yl < yh; else raise ValueError.
    • nx ≥ 1 and ny ≥ 1; else raise ValueError.
    Grid, node IDs, and coordinates
    • Define dx = (xh − xl)/nx and dy = (yh − yl)/ny.
    • Construct a refined grid with dimensions npx = 2*nx+1 and npy = 2*ny+1.
      Each refined step is 0.5*dx in x and 0.5*dy in y.
    • Central grid points located at odd (ix, iy) pairs (i.e., (ix0+1, iy0+1) in each
      2×2 refined block corresponding to one coarse cell) are excluded from the global
      node set (these are geometric cell centers).
    • Global node IDs are zero-based and assigned in row-major order with x varying fastest:
        for iy = 0..(npy−1): for ix = 0..(npx−1):
            if not (ix%2==1 and iy%2==1):
                id ← next integer
                coords[id] = [ xl + 0.5*dx*ix , yl + 0.5*dy*iy ]  (float64)
      Equivalently: coordinates come from meshgrid(indexing="xy") on the refined axes,
      flattened in C-order, with centers removed. No coordinates are computed by averaging.
    Cell traversal and element connectivity
    • Traverse cells row-major: cy = 0..ny−1 (bottom→top), and for each cy, cx = 0..nx−1 (left→right).
    • Each cell (cx, cy) corresponds to refined grid origin (ix0, iy0) = (2*cx, 2*cy).
    • Emit exactly one Quad8 element per cell with node ordering:
        [N1, N2, N3, N4, N5, N6, N7, N8], where
          N1 = (ix0,   iy0)     bottom-left corner
          N2 = (ix0+2, iy0)     bottom-right corner
          N3 = (ix0+2, iy0+2)   top-right corner
          N4 = (ix0,   iy0+2)   top-left corner
          N5 = (ix0+1, iy0)     midside bottom   (N1–N2)
          N6 = (ix0+2, iy0+1)   midside right    (N2–N3)
          N7 = (ix0+1, iy0+2)   midside top      (N3–N4)
          N8 = (ix0,   iy0+1)   midside left     (N4–N1)
      Do not reorder elements after emission. All midside nodes reference the refined grid
      node IDs (shared across neighboring elements), not per-element duplicates.
    Types and shapes
    • coords is an (Nnodes, 2) ndarray, dtype float64, where
        Nnodes = (2*nx+1)*(2*ny+1) − nx*ny   (refined grid minus excluded centers).
    • connect is an (Ne, 8) ndarray, dtype int64, where Ne = nx*ny.
      Each row lists the 8 node IDs in the order [N1..N8] defined above.
    Parameters
    ----------
    xl, yl, xh, yh : float
        Domain bounds with xl < xh and yl < yh.
    nx, ny : int
        Number of rectangular subdivisions in x and y (each ≥ 1).
    Returns
    -------
    coords : (Nnodes, 2) float64 ndarray
        Node coordinates in row-major (x-fastest) order, excluding central nodes.
    connect : (nx*ny, 8) int64 ndarray
        Quad8 connectivity with rows [N1..N8] as specified above.
    Raises
    ------
    ValueError
        If nx < 1 or ny < 1, or if xl ≥ xh or yl ≥ yh.
    Notes
    -----
    • Quad8 includes 4 corners + 4 midsides; the cell center node is intentionally omitted.
    • The mesh is conforming: shared corners/edges reuse identical global node IDs.
    • Following this contract guarantees bit-for-bit identical outputs across implementations.
    """
    if xl >= xh or yl >= yh:
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh')
    if nx < 1 or ny < 1:
        raise ValueError('nx and ny must be ≥ 1')
    dx = (xh - xl) / nx
    dy = (yh - yl) / ny
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    x_vals = np.linspace(xl, xh, npx, dtype=np.float64)
    y_vals = np.linspace(yl, yh, npy, dtype=np.float64)
    node_id_map = np.full((npx, npy), -1, dtype=np.int64)
    node_count = 0
    for iy in range(npy):
        for ix in range(npx):
            if not (ix % 2 == 1 and iy % 2 == 1):
                node_id_map[ix, iy] = node_count
                node_count += 1
    coords = np.zeros((node_count, 2), dtype=np.float64)
    for iy in range(npy):
        for ix in range(npx):
            node_id = node_id_map[ix, iy]
            if node_id != -1:
                coords[node_id] = [x_vals[ix], y_vals[iy]]
    ne = nx * ny
    connect = np.zeros((ne, 8), dtype=np.int64)
    for cy in range(ny):
        for cx in range(nx):
            elem_id = cy * nx + cx
            ix0 = 2 * cx
            iy0 = 2 * cy
            n1 = node_id_map[ix0, iy0]
            n2 = node_id_map[ix0 + 2, iy0]
            n3 = node_id_map[ix0 + 2, iy0 + 2]
            n4 = node_id_map[ix0, iy0 + 2]
            n5 = node_id_map[ix0 + 1, iy0]
            n6 = node_id_map[ix0 + 2, iy0 + 1]
            n7 = node_id_map[ix0 + 1, iy0 + 2]
            n8 = node_id_map[ix0, iy0 + 1]
            connect[elem_id] = [n1, n2, n3, n4, n5, n6, n7, n8]
    return (coords, connect)