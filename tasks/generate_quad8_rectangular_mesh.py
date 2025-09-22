import numpy as np
import pytest
from typing import Tuple


def generate_quad8_rectangular_mesh(
    xl: float,
    yl: float,
    xh: float,
    yh: float,
    nx: int,
    ny: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    # --- Preconditions (validate before any divisions) ---
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be ≥ 1")
    if not (xl < xh and yl < yh):
        raise ValueError("Domain bounds must satisfy xl < xh and yl < yh")

    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)
    npx = 2 * nx + 1
    npy = 2 * ny + 1

    # Build refined coordinates in row-major order (x-fastest), skipping centers.
    node_map: dict[tuple[int, int], int] = {}
    coords_list: list[tuple[float, float]] = []
    new_index = 0
    for iy in range(npy):          # bottom → top
        y = yl + 0.5 * dy * iy
        for ix in range(npx):      # left → right
            if ix % 2 == 1 and iy % 2 == 1:  # skip central nodes
                continue
            x = xl + 0.5 * dx * ix
            node_map[(ix, iy)] = new_index
            coords_list.append((x, y))
            new_index += 1

    coords = np.array(coords_list, dtype=np.float64)

    def node_id(ix: int, iy: int) -> int:
        return node_map[(ix, iy)]

    # Assemble connectivity with the exact Quad8 ordering per cell.
    connect = np.empty((nx * ny, 8), dtype=np.int64)
    e = 0
    for cy in range(ny):
        for cx in range(nx):
            ix0 = 2 * cx
            iy0 = 2 * cy
            connect[e, 0] = node_id(ix0,     iy0)     # N1: BL
            connect[e, 1] = node_id(ix0 + 2, iy0)     # N2: BR
            connect[e, 2] = node_id(ix0 + 2, iy0 + 2) # N3: TR
            connect[e, 3] = node_id(ix0,     iy0 + 2) # N4: TL
            connect[e, 4] = node_id(ix0 + 1, iy0)     # N5: bottom mid
            connect[e, 5] = node_id(ix0 + 2, iy0 + 1) # N6: right mid
            connect[e, 6] = node_id(ix0 + 1, iy0 + 2) # N7: top mid
            connect[e, 7] = node_id(ix0,     iy0 + 1) # N8: left mid
            e += 1

    return coords, connect


def generate_quad8_rectangular_mesh_fullstep_bug(
    xl: float, yl: float, xh: float, yh: float, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be ≥ 1")
    if not (xl < xh and yl < yh):
        raise ValueError("Domain bounds must satisfy xl < xh and yl < yh")

    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)
    npx, npy = 2*nx + 1, 2*ny + 1

    # BUG: use full-step spacing instead of half-step
    xs = xl + dx * np.arange(npx, dtype=np.float64)   # ❌ should be 0.5*dx
    ys = yl + dy * np.arange(npy, dtype=np.float64)   # ❌

    XX, YY = np.meshgrid(xs, ys, indexing="xy")

    node_map = {}
    coords_list = []
    idx = 0
    for iy in range(npy):
        for ix in range(npx):
            if ix % 2 == 1 and iy % 2 == 1:  # skip centers (correct rule)
                continue
            node_map[(ix, iy)] = idx
            coords_list.append((XX[iy, ix], YY[iy, ix]))
            idx += 1
    coords = np.array(coords_list, dtype=np.float64)

    def node_id(ix: int, iy: int) -> int:
        return node_map[(ix, iy)]

    connect = np.empty((nx*ny, 8), dtype=np.int64)
    e = 0
    for cy in range(ny):
        for cx in range(nx):
            ix0, iy0 = 2*cx, 2*cy
            connect[e, 0] = node_id(ix0,     iy0)
            connect[e, 1] = node_id(ix0+2,   iy0)
            connect[e, 2] = node_id(ix0+2,   iy0+2)
            connect[e, 3] = node_id(ix0,     iy0+2)
            connect[e, 4] = node_id(ix0+1,   iy0)
            connect[e, 5] = node_id(ix0+2,   iy0+1)
            connect[e, 6] = node_id(ix0+1,   iy0+2)
            connect[e, 7] = node_id(ix0,     iy0+1)
            e += 1
    return coords, connect


def generate_quad8_rectangular_mesh_ccw_bug(
    xl: float, yl: float, xh: float, yh: float, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be ≥ 1")
    if not (xl < xh and yl < yh):
        raise ValueError("Domain bounds must satisfy xl < xh and yl < yh")

    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)
    npx, npy = 2 * nx + 1, 2 * ny + 1

    # Build refined coordinates (x-fastest), skipping central nodes (odd,odd)
    node_map = {}
    coords_list = []
    idx = 0
    for iy in range(npy):
        y = yl + 0.5 * dy * iy
        for ix in range(npx):
            if ix % 2 == 1 and iy % 2 == 1:
                continue
            x = xl + 0.5 * dx * ix
            node_map[(ix, iy)] = idx
            coords_list.append((x, y))
            idx += 1
    coords = np.array(coords_list, dtype=np.float64)

    def node_id(ix: int, iy: int) -> int:
        return node_map[(ix, iy)]

    # Assemble connectivity with a CLOCKWISE corner ordering bug:
    # Corners should be [BL, BR, TR, TL] (CCW). We swap N1<->N2 to make it CW on purpose.
    connect = np.empty((nx * ny, 8), dtype=np.int64)
    e = 0
    for cy in range(ny):
        for cx in range(nx):
            ix0, iy0 = 2 * cx, 2 * cy
            # BUG: corners clockwise: [BR, BL, TL, TR]
            connect[e, 0] = node_id(ix0 + 2, iy0)       # N1: should be BL
            connect[e, 1] = node_id(ix0,     iy0)       # N2: should be BR
            connect[e, 2] = node_id(ix0,     iy0 + 2)   # N3: should be TR
            connect[e, 3] = node_id(ix0 + 2, iy0 + 2)   # N4: should be TL
            # Keep midsides in the nominal positions (bottom, right, top, left)
            connect[e, 4] = node_id(ix0 + 1, iy0)       # N5: bottom mid
            connect[e, 5] = node_id(ix0 + 2, iy0 + 1)   # N6: right mid
            connect[e, 6] = node_id(ix0 + 1, iy0 + 2)   # N7: top mid
            connect[e, 7] = node_id(ix0,     iy0 + 1)   # N8: left mid
            e += 1

    return coords, connect


def generate_quad8_rectangular_mesh_duplicate_node_bug(
    xl: float, yl: float, xh: float, yh: float, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be ≥ 1")
    if not (xl < xh and yl < yh):
        raise ValueError("Domain bounds must satisfy xl < xh and yl < yh")

    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)
    npx, npy = 2 * nx + 1, 2 * ny + 1

    # Build refined coordinates (x-fastest), skipping central nodes (odd,odd)
    node_map = {}
    coords_list = []
    idx = 0
    for iy in range(npy):
        y = yl + 0.5 * dy * iy
        for ix in range(npx):
            if ix % 2 == 1 and iy % 2 == 1:
                continue
            x = xl + 0.5 * dx * ix
            node_map[(ix, iy)] = idx
            coords_list.append((x, y))
            idx += 1
    coords = np.array(coords_list, dtype=np.float64)

    def node_id(ix: int, iy: int) -> int:
        return node_map[(ix, iy)]

    # Assemble connectivity with a deliberate duplicate node bug:
    # N5 (bottom midside) incorrectly duplicates N1 (bottom-left corner).
    connect = np.empty((nx * ny, 8), dtype=np.int64)
    e = 0
    for cy in range(ny):
        for cx in range(nx):
            ix0, iy0 = 2 * cx, 2 * cy
            N1 = node_id(ix0,     iy0)       # BL
            N2 = node_id(ix0 + 2, iy0)       # BR
            N3 = node_id(ix0 + 2, iy0 + 2)   # TR
            N4 = node_id(ix0,     iy0 + 2)   # TL

            # BUG: N5 should be node_id(ix0+1, iy0) but we duplicate N1 instead
            N5 = N1                           # bottom midside (incorrect)
            N6 = node_id(ix0 + 2, iy0 + 1)    # right mid
            N7 = node_id(ix0 + 1, iy0 + 2)    # top mid
            N8 = node_id(ix0,     iy0 + 1)    # left mid

            connect[e, :] = (N1, N2, N3, N4, N5, N6, N7, N8)
            e += 1

    return coords, connect


def generate_quad8_rectangular_mesh_no_error_nx_ny(
    xl: float, yl: float, xh: float, yh: float, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUGGY (for testing): silently accept nx<=0 or ny<=0 by clamping to 1.
    Produces a mesh instead of raising ValueError, violating the spec.
    """
    # --- BUG: silently clamp instead of raising ---
    nx_eff = max(1, int(nx))
    ny_eff = max(1, int(ny))

    if not (xl < xh and yl < yh):
        raise ValueError("Domain bounds must satisfy xl < xh and yl < yh")

    dx = (xh - xl) / float(nx_eff)
    dy = (yh - yl) / float(ny_eff)
    npx, npy = 2 * nx_eff + 1, 2 * ny_eff + 1

    # Build refined coordinates (x-fastest), skipping cell centers (odd,odd)
    node_map = {}
    coords_list = []
    idx = 0
    for iy in range(npy):
        y = yl + 0.5 * dy * iy
        for ix in range(npx):
            if ix % 2 == 1 and iy % 2 == 1:
                continue
            x = xl + 0.5 * dx * ix
            node_map[(ix, iy)] = idx
            coords_list.append((x, y))
            idx += 1
    coords = np.array(coords_list, dtype=np.float64)

    def node_id(ix: int, iy: int) -> int:
        return node_map[(ix, iy)]

    # Quad8 connectivity (one element per coarse cell)
    connect = np.empty((nx_eff * ny_eff, 8), dtype=np.int64)
    e = 0
    for cy in range(ny_eff):
        for cx in range(nx_eff):
            ix0, iy0 = 2 * cx, 2 * cy
            connect[e, 0] = node_id(ix0,     iy0)     # N1 BL
            connect[e, 1] = node_id(ix0 + 2, iy0)     # N2 BR
            connect[e, 2] = node_id(ix0 + 2, iy0 + 2) # N3 TR
            connect[e, 3] = node_id(ix0,     iy0 + 2) # N4 TL
            connect[e, 4] = node_id(ix0 + 1, iy0)     # N5 bottom mid
            connect[e, 5] = node_id(ix0 + 2, iy0 + 1) # N6 right mid
            connect[e, 6] = node_id(ix0 + 1, iy0 + 2) # N7 top mid
            connect[e, 7] = node_id(ix0,     iy0 + 1) # N8 left mid
            e += 1

    return coords, connect


def generate_quad8_rectangular_mesh_no_error_domain(
    xl: float, yl: float, xh: float, yh: float, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUGGY (for testing): silently accept invalid domain extents by swapping.
    If xl >= xh or yl >= yh, swaps the bounds instead of raising ValueError,
    violating the spec while still producing a mesh.
    """
    # --- BUG: silently swap invalid extents instead of raising ---
    if xl >= xh:
        xl, xh = xh, xl
    if yl >= yh:
        yl, yh = yh, yl

    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be ≥ 1")

    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)
    npx, npy = 2 * nx + 1, 2 * ny + 1

    # Build refined coordinates (x-fastest), skipping cell centers (odd,odd)
    node_map = {}
    coords_list = []
    idx = 0
    for iy in range(npy):
        y = yl + 0.5 * dy * iy
        for ix in range(npx):
            if ix % 2 == 1 and iy % 2 == 1:
                continue
            x = xl + 0.5 * dx * ix
            node_map[(ix, iy)] = idx
            coords_list.append((x, y))
            idx += 1
    coords = np.array(coords_list, dtype=np.float64)

    def node_id(ix: int, iy: int) -> int:
        return node_map[(ix, iy)]

    # Quad8 connectivity (one element per coarse cell)
    connect = np.empty((nx * ny, 8), dtype=np.int64)
    e = 0
    for cy in range(ny):
        for cx in range(nx):
            ix0, iy0 = 2 * cx, 2 * cy
            connect[e, 0] = node_id(ix0,     iy0)     # N1 BL
            connect[e, 1] = node_id(ix0 + 2, iy0)     # N2 BR
            connect[e, 2] = node_id(ix0 + 2, iy0 + 2) # N3 TR
            connect[e, 3] = node_id(ix0,     iy0 + 2) # N4 TL
            connect[e, 4] = node_id(ix0 + 1, iy0)     # N5 bottom mid
            connect[e, 5] = node_id(ix0 + 2, iy0 + 1) # N6 right mid
            connect[e, 6] = node_id(ix0 + 1, iy0 + 2) # N7 top mid
            connect[e, 7] = node_id(ix0,     iy0 + 1) # N8 left mid
            e += 1

    return coords, connect


def test_quad8_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain for Quad8 elements.

    Checks:
    - Correct number of nodes (refined grid minus one skipped center per cell) and elements.
    - Array shapes and dtypes are as expected.
    - Corner nodes coincide with the domain boundaries (found by coordinate match).
    - Node coordinates form a uniform half-step lattice (dx/2, dy/2).
    - Repeated calls yield identical results (determinism).
    """
    xl, yl, xh, yh = 0.0, 0.0, 1.0, 1.0
    nx, ny = 2, 2

    coords1, connect1 = fcn(xl, yl, xh, yh, nx, ny)
    coords2, connect2 = fcn(xl, yl, xh, yh, nx, ny)

    # Sizes: refined grid has (2*nx+1)*(2*ny+1) points; skip nx*ny cell centers
    npx, npy = 2*nx + 1, 2*ny + 1
    expected_nodes = npx * npy - nx * ny
    expected_elems = nx * ny

    assert coords1.shape == (expected_nodes, 2)
    assert connect1.shape == (expected_elems, 8)
    assert coords1.dtype == np.float64
    assert connect1.dtype in (np.int64, np.int32)

    # Corner coordinates must exist exactly once each
    corners = np.array([[xl, yl], [xh, yl], [xl, yh], [xh, yh]], dtype=np.float64)
    for cx, cy in corners:
        matches = np.where(np.isclose(coords1[:, 0], cx) & np.isclose(coords1[:, 1], cy))[0]
        assert matches.size == 1, f"Corner ({cx},{cy}) not found exactly once."

    # Uniform half-step lattice across the whole domain
    dx, dy = (xh - xl)/nx, (yh - yl)/ny
    xs, ys = np.unique(coords1[:, 0]), np.unique(coords1[:, 1])
    assert np.isclose(xs.min(), xl) and np.isclose(xs.max(), xh)
    assert np.isclose(ys.min(), yl) and np.isclose(ys.max(), yh)
    assert np.allclose(np.diff(xs), 0.5*dx)
    assert np.allclose(np.diff(ys), 0.5*dy)

    # Determinism
    assert np.array_equal(connect1, connect2)
    assert np.allclose(coords1, coords2)


def test_quad8_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain for Quad8 elements.

    Checks:
    - Connectivity indices are within valid range and unique per element.
    - Corner nodes (N1–N4) are consistently counter-clockwise (positive polygon area).
    - Midside nodes (N5–N8) equal the average of their adjacent corner nodes.
    - Shared edges between elements reuse identical corner node IDs (conforming mesh).
    """
    xl, yl, xh, yh = -1.0, 2.0, 3.0, 5.0
    nx, ny = 3, 1
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)

    # --- Index validity and per-element uniqueness ---
    n_nodes = coords.shape[0]
    assert connect.min() >= 0 and connect.max() < n_nodes
    for e in connect:
        assert len(set(e.tolist())) == 8  # no duplicates within an element

    # --- Corner orientation: N1..N4 must be CCW (positive signed polygon area) ---
    # Quad8 ordering: N1=BL, N2=BR, N3=TR, N4=TL
    for e in connect:
        p = coords[e[:4]]  # shape (4,2)
        area2 = (
            p[0,0]*p[1,1] - p[0,1]*p[1,0] +
            p[1,0]*p[2,1] - p[1,1]*p[2,0] +
            p[2,0]*p[3,1] - p[2,1]*p[3,0] +
            p[3,0]*p[0,1] - p[3,1]*p[0,0]
        )
        assert area2 > 0.0  # CCW

    # --- Midside placement: arithmetic means of adjacent corners ---
    # N5 on (N1,N2), N6 on (N2,N3), N7 on (N3,N4), N8 on (N4,N1)
    c = coords
    atol = 1e-12
    assert np.allclose(c[connect[:, 4]], 0.5*(c[connect[:, 0]] + c[connect[:, 1]]), atol=atol)  # bottom mid
    assert np.allclose(c[connect[:, 5]], 0.5*(c[connect[:, 1]] + c[connect[:, 2]]), atol=atol)  # right mid
    assert np.allclose(c[connect[:, 6]], 0.5*(c[connect[:, 2]] + c[connect[:, 3]]), atol=atol)  # top mid
    assert np.allclose(c[connect[:, 7]], 0.5*(c[connect[:, 3]] + c[connect[:, 0]]), atol=atol)  # left mid

    # --- Conformity: shared edges (by corner IDs) appear exactly once (boundary) or twice (interior) ---
    edges = []
    for e in connect:
        edges.extend([
            tuple(sorted((e[0], e[1]))),  # bottom edge by corners
            tuple(sorted((e[1], e[2]))),  # right edge by corners
            tuple(sorted((e[2], e[3]))),  # top edge by corners
            tuple(sorted((e[3], e[0]))),  # left edge by corners
        ])

    # Each corner-defined edge must occur at most twice; at least one interior edge should occur twice
    unique_edges = set(edges)
    for edge in unique_edges:
        count = edges.count(edge)
        assert count in (1, 2)
    assert any(edges.count(edge) == 2 for edge in unique_edges)


def test_quad8_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs in Quad8 mesh generation.

    Checks:
    - nx <= 0 raises ValueError.
    - ny <= 0 raises ValueError.
    - xl >= xh raises ValueError.
    - yl >= yh raises ValueError.
    """
    # Invalid subdivisions
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 0, 1)
    with pytest.raises(ValueError):
        fcn(0.0, 0.0, 1.0, 1.0, 1, 0)

    # Invalid domain extents
    with pytest.raises(ValueError):
        fcn(1.0, 0.0, 0.0, 1.0, 2, 2)  # xl >= xh
    with pytest.raises(ValueError):
        fcn(0.0, 2.0, 1.0, 1.0, 2, 2)  # yl >= yh


def task_info():
    task_id = "generate_quad8_rectangular_mesh"
    task_short_description = "generates a rectangular mesh of quad8 elements"
    created_date = "2025-09-22"
    created_by = "elejeune11"
    main_fcn = generate_quad8_rectangular_mesh
    required_imports = ["import numpy as np", "import pytest", "from typing import Tuple"]
    fcn_dependencies = []
    reference_verification_inputs = [
        # 1) Minimal case on unit square
        [0.0, 0.0, 1.0, 1.0, 1, 1],
        # 2) Shifted domain with negative coordinates
        [-2.5, 3.0, 0.5, 7.0, 3, 2],
        # 3) Rectangular domain with aspect ratio and moderate refinement
        [0.0, 0.0, 2.0, 1.0, 4, 3],
        # 4) Small extent domain
        [10.0, -5.0, 10.6, -4.2, 1, 4],
        # 5) Micro-scale domain with higher refinement
        [1e-6, 1e-6, 1e-3, 2e-3, 5, 7],
    ]
    test_cases = [{"test_code": test_quad8_mesh_basic_structure_and_determinism, "expected_failures": [generate_quad8_rectangular_mesh_fullstep_bug]},
                  {"test_code": test_quad8_mesh_geometry_and_conformity, "expected_failures": [generate_quad8_rectangular_mesh_ccw_bug, generate_quad8_rectangular_mesh_duplicate_node_bug]},
                  {"test_code": test_quad8_mesh_invalid_inputs, "expected_failures": [generate_quad8_rectangular_mesh_no_error_nx_ny, generate_quad8_rectangular_mesh_no_error_domain]}
                  ]
    return task_id, task_short_description, created_date, created_by, main_fcn, required_imports, fcn_dependencies, reference_verification_inputs, test_cases
