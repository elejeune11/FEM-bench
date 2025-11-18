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
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh.')
    if nx < 1 or ny < 1:
        raise ValueError('Number of subdivisions nx and ny must be at least 1.')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    x_coords_refined = np.linspace(xl, xh, npx, dtype=np.float64)
    y_coords_refined = np.linspace(yl, yh, npy, dtype=np.float64)
    (xx, yy) = np.meshgrid(x_coords_refined, y_coords_refined, indexing='xy')
    (ix_grid, iy_grid) = np.meshgrid(np.arange(npx), np.arange(npy), indexing='xy')
    is_center_mask = (ix_grid % 2 == 1) & (iy_grid % 2 == 1)
    is_node_mask = ~is_center_mask
    coords = np.vstack([xx.ravel(), yy.ravel()]).T[is_node_mask.ravel()]
    id_map = np.full((npy, npx), -1, dtype=np.int64)
    id_map.ravel()[is_node_mask.ravel()] = np.arange(np.sum(is_node_mask))
    (cx_all, cy_all) = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    cx_flat = cx_all.ravel()
    cy_flat = cy_all.ravel()
    ix0 = 2 * cx_flat
    iy0 = 2 * cy_flat
    node_offsets = [(0, 0), (2, 0), (2, 2), (0, 2), (1, 0), (2, 1), (1, 2), (0, 1)]
    connectivity_nodes = []
    for (dx_offset, dy_offset) in node_offsets:
        ix = ix0 + dx_offset
        iy = iy0 + dy_offset
        node_ids = id_map[iy, ix]
        connectivity_nodes.append(node_ids)
    connect = np.stack(connectivity_nodes, axis=1)
    return (coords, connect)