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
    if not (xl < xh and yl < yh):
        raise ValueError('Domain bounds must satisfy xl < xh and yl < yh.')
    if not (nx >= 1 and ny >= 1):
        raise ValueError('Number of subdivisions nx and ny must be at least 1.')
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    x_coords_refined = np.linspace(xl, xh, npx, dtype=np.float64)
    y_coords_refined = np.linspace(yl, yh, npy, dtype=np.float64)
    (xx, yy) = np.meshgrid(x_coords_refined, y_coords_refined, indexing='xy')
    all_coords = np.stack((xx, yy), axis=-1)
    ix_indices = np.arange(npx)
    iy_indices = np.arange(npy)
    is_center_mask = (ix_indices % 2 == 1)[None, :] & (iy_indices % 2 == 1)[:, None]
    is_valid_node_mask = ~is_center_mask
    coords = all_coords[is_valid_node_mask]
    num_nodes = is_valid_node_mask.sum()
    id_map = np.full((npy, npx), -1, dtype=np.int64)
    id_map[is_valid_node_mask] = np.arange(num_nodes, dtype=np.int64)
    (cx_grid, cy_grid) = np.meshgrid(np.arange(nx), np.arange(ny), indexing='xy')
    ix0 = (2 * cx_grid).flatten()
    iy0 = (2 * cy_grid).flatten()
    local_ix_offsets = np.array([0, 2, 2, 0, 1, 2, 1, 0], dtype=np.int64)
    local_iy_offsets = np.array([0, 0, 2, 2, 0, 1, 2, 1], dtype=np.int64)
    global_ix = ix0[:, None] + local_ix_offsets[None, :]
    global_iy = iy0[:, None] + local_iy_offsets[None, :]
    connect = id_map[global_iy, global_ix]
    return (coords, connect)