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
    import pytest
    from typing import Tuple as _Tuple
    if nx < 1 or ny < 1 or (not xl < xh) or (not yl < yh):
        raise ValueError('Invalid domain bounds or subdivision counts.')
    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1
    Nnodes = npx * npy
    Ne = 2 * nx * ny
    ix = np.arange(npx, dtype=np.int64)
    iy = np.arange(npy, dtype=np.int64)
    (IX, IY) = np.meshgrid(ix, iy, indexing='xy')
    coords = np.empty((Nnodes, 2), dtype=np.float64)
    coords[:, 0] = xl + 0.5 * dx * IX.ravel(order='C')
    coords[:, 1] = yl + 0.5 * dy * IY.ravel(order='C')

    def node_id(ixi: int, iyi: int) -> int:
        return int(iyi * npx + ixi)
    connect = np.empty((Ne, 6), dtype=np.int64)
    e = 0
    for cy in range(ny):
        iyB = 2 * cy
        iyT = iyB + 2
        iyM = iyB + 1
        for cx in range(nx):
            ixL = 2 * cx
            ixR = ixL + 2
            ixM = ixL + 1
            bl = node_id(ixL, iyB)
            br = node_id(ixR, iyB)
            tl = node_id(ixL, iyT)
            tr = node_id(ixR, iyT)
            midBottom = node_id(ixM, iyB)
            midTop = node_id(ixM, iyT)
            midLeft = node_id(ixL, iyM)
            midRight = node_id(ixR, iyM)
            midDiag = node_id(ixM, iyM)
            connect[e, 0] = br
            connect[e, 1] = tl
            connect[e, 2] = bl
            connect[e, 3] = midDiag
            connect[e, 4] = midLeft
            connect[e, 5] = midBottom
            e += 1
            connect[e, 0] = tr
            connect[e, 1] = tl
            connect[e, 2] = br
            connect[e, 3] = midTop
            connect[e, 4] = midDiag
            connect[e, 5] = midRight
            e += 1
    return (coords, connect)