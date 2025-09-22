import numpy as np
import pytest
from typing import Tuple


def generate_tri6_rectangular_mesh(
    xl: float,
    yl: float,
    xh: float,
    yh: float,
    nx: int,
    ny: int
) -> Tuple[np.ndarray, np.ndarray]:
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
    # --- Preconditions for reproducibility and well-posedness ---
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be ≥ 1")
    if not (xl < xh and yl < yh):
        raise ValueError("Domain extents must satisfy xl < xh and yl < yh")

    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)

    # Refined grid: step = 0.5*dx, 0.5*dy; nodes numbered row-major with x fastest.
    npx, npy = 2 * nx + 1, 2 * ny + 1
    xs = xl + 0.5 * dx * np.arange(npx, dtype=np.float64)
    ys = yl + 0.5 * dy * np.arange(npy, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    coords = np.column_stack([XX.ravel(order="C"), YY.ravel(order="C")]).astype(np.float64, copy=False)

    def node_id(ix: int, iy: int) -> int:
        return iy * npx + ix  # row-major, x fastest

    connect = np.empty((2 * nx * ny, 6), dtype=np.int64)
    e = 0

    for cy in range(ny):
        for cx in range(nx):
            ix0 = 2 * cx
            iy0 = 2 * cy

            # First triangle: (br, tl, bl) CCW
            N1 = node_id(ix0 + 2, iy0    )  # bottom-right
            N2 = node_id(ix0,     iy0 + 2)  # top-left
            N3 = node_id(ix0,     iy0    )  # bottom-left
            N4 = node_id(ix0 + 1, iy0 + 1)  # midside (N1–N2)
            N5 = node_id(ix0,     iy0 + 1)  # midside (N2–N3)
            N6 = node_id(ix0 + 1, iy0    )  # midside (N3–N1)
            connect[e, :] = (N1, N2, N3, N4, N5, N6); e += 1

            # Second triangle: (tr, tl, br) CCW
            N1 = node_id(ix0 + 2, iy0 + 2)  # top-right
            N2 = node_id(ix0,     iy0 + 2)  # top-left
            N3 = node_id(ix0 + 2, iy0    )  # bottom-right
            N4 = node_id(ix0 + 1, iy0 + 2)  # midside (N1–N2)
            N5 = node_id(ix0 + 1, iy0 + 1)  # midside (N2–N3)
            N6 = node_id(ix0 + 2, iy0 + 1)  # midside (N3–N1)
            connect[e, :] = (N1, N2, N3, N4, N5, N6); e += 1

    return coords, connect


def generate_tri6_rectangular_mesh_indexing_bug(
    xl: float, yl: float, xh: float, yh: float, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be ≥ 1")

    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)

    npx, npy = 2 * nx + 1, 2 * ny + 1
    xs = xl + 0.5 * dx * np.arange(npx, dtype=np.float64)
    ys = yl + 0.5 * dy * np.arange(npy, dtype=np.float64)

    # BUG: wrong meshgrid indexing; rest of code assumes "xy"
    XX, YY = np.meshgrid(xs, ys, indexing="ij")  # <-- should be "xy"
    coords = np.column_stack([XX.ravel(), YY.ravel()])

    def node_id(ix: int, iy: int) -> int:
        return iy * npx + ix

    connect = np.empty((2 * nx * ny, 6), dtype=np.int64)
    e = 0
    for cy in range(ny):
        for cx in range(nx):
            ix0, iy0 = 2 * cx, 2 * cy

            # First triangle
            N1 = node_id(ix0 + 2, iy0)      # bottom-right
            N2 = node_id(ix0,     iy0 + 2)  # top-left
            N3 = node_id(ix0,     iy0)      # bottom-left
            N4 = node_id(ix0 + 1, iy0 + 1)  # diag midpoint (N1–N2)
            N5 = node_id(ix0,     iy0 + 1)  # left midpoint (N2–N3)
            N6 = node_id(ix0 + 1, iy0)      # bottom midpoint (N3–N1)
            connect[e] = [N1, N2, N3, N4, N5, N6]; e += 1

            # Second triangle
            N1 = node_id(ix0 + 2, iy0 + 2)  # top-right
            N2 = node_id(ix0,     iy0 + 2)  # top-left
            N3 = node_id(ix0 + 2, iy0)      # bottom-right
            N4 = node_id(ix0 + 1, iy0 + 2)  # top midpoint (N1–N2)
            N5 = node_id(ix0 + 1, iy0 + 1)  # diag midpoint (N2–N3)
            N6 = node_id(ix0 + 2, iy0 + 1)  # right midpoint (N3–N1)
            connect[e] = [N1, N2, N3, N4, N5, N6]; e += 1

    return coords, connect


def generate_tri6_rectangular_mesh_ccw_bug(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        # Keep same behavior as the good function to avoid exceptions in “normal” runs
        raise ValueError("nx and ny must be ≥ 1")
    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)

    npx, npy = 2 * nx + 1, 2 * ny + 1
    xs = xl + 0.5 * dx * np.arange(npx, dtype=np.float64)
    ys = yl + 0.5 * dy * np.arange(npy, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    coords = np.column_stack([XX.ravel(), YY.ravel()])

    def node_id(ix: int, iy: int) -> int:
        return iy * npx + ix

    connect = np.empty((2 * nx * ny, 6), dtype=np.int64)
    e = 0
    for cy in range(ny):
        for cx in range(nx):
            ix0, iy0 = 2 * cx, 2 * cy

            # BUG: swap N1<->N2 makes this triangle clockwise
            N1 = node_id(ix0,     iy0 + 2)  # should be bottom-right
            N2 = node_id(ix0 + 2, iy0)      # should be top-left
            N3 = node_id(ix0,     iy0)
            N4 = node_id(ix0 + 1, iy0 + 1)
            N5 = node_id(ix0,     iy0 + 1)
            N6 = node_id(ix0 + 1, iy0)
            connect[e] = [N1, N2, N3, N4, N5, N6]; e += 1

            # Second triangle unchanged (still CCW)
            N1 = node_id(ix0 + 2, iy0 + 2)
            N2 = node_id(ix0,     iy0 + 2)
            N3 = node_id(ix0 + 2, iy0)
            N4 = node_id(ix0 + 1, iy0 + 2)
            N5 = node_id(ix0 + 1, iy0 + 1)
            N6 = node_id(ix0 + 2, iy0 + 1)
            connect[e] = [N1, N2, N3, N4, N5, N6]; e += 1

    return coords, connect


def generate_tri6_rectangular_mesh_duplicate_node_bug(xl: float, yl: float, xh: float, yh: float, nx: int, ny: int) -> Tuple[np.ndarray, np.ndarray]:
    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be ≥ 1")
    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)

    npx, npy = 2 * nx + 1, 2 * ny + 1
    xs = xl + 0.5 * dx * np.arange(npx, dtype=np.float64)
    ys = yl + 0.5 * dy * np.arange(npy, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    coords = np.column_stack([XX.ravel(), YY.ravel()])

    def node_id(ix: int, iy: int) -> int:
        return iy * npx + ix

    connect = np.empty((2 * nx * ny, 6), dtype=np.int64)
    e = 0
    for cy in range(ny):
        for cx in range(nx):
            ix0, iy0 = 2 * cx, 2 * cy

            # First triangle (CCW corners ok) but BUG: N4 duplicates N1
            N1 = node_id(ix0 + 2, iy0)
            N2 = node_id(ix0,     iy0 + 2)
            N3 = node_id(ix0,     iy0)
            N4 = N1                      # BUG: duplicate node id
            N5 = node_id(ix0,     iy0 + 1)
            N6 = node_id(ix0 + 1, iy0)
            connect[e] = [N1, N2, N3, N4, N5, N6]; e += 1

            # Second triangle correct
            N1 = node_id(ix0 + 2, iy0 + 2)
            N2 = node_id(ix0,     iy0 + 2)
            N3 = node_id(ix0 + 2, iy0)
            N4 = node_id(ix0 + 1, iy0 + 2)
            N5 = node_id(ix0 + 1, iy0 + 1)
            N6 = node_id(ix0 + 2, iy0 + 1)
            connect[e] = [N1, N2, N3, N4, N5, N6]; e += 1

    return coords, connect


def generate_tri6_rectangular_mesh_no_error_nx_ny(
    xl: float, yl: float, xh: float, yh: float, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUGGY (for testing): silently accept nx<=0 or ny<=0 by clamping to 1.
    This avoids division-by-zero and produces a mesh, but violates the spec
    that requires raising ValueError for invalid inputs.
    """
    # --- BUG: silently clamp instead of raising ---
    nx_eff = max(1, int(nx))
    ny_eff = max(1, int(ny))

    dx = (xh - xl) / float(nx_eff)
    dy = (yh - yl) / float(ny_eff)

    npx, npy = 2 * nx_eff + 1, 2 * ny_eff + 1
    xs = xl + 0.5 * dx * np.arange(npx, dtype=np.float64)
    ys = yl + 0.5 * dy * np.arange(npy, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    coords = np.column_stack([XX.ravel(), YY.ravel()])  # (Nnodes, 2) float64

    def node_id(ix: int, iy: int) -> int:
        return iy * npx + ix

    connect = np.empty((2 * nx_eff * ny_eff, 6), dtype=np.int64)
    e = 0
    for cy in range(ny_eff):
        for cx in range(nx_eff):
            ix0 = 2 * cx
            iy0 = 2 * cy

            # First triangle (CCW)
            N1 = node_id(ix0 + 2, iy0)
            N2 = node_id(ix0,     iy0 + 2)
            N3 = node_id(ix0,     iy0)
            N4 = node_id(ix0 + 1, iy0 + 1)
            N5 = node_id(ix0,     iy0 + 1)
            N6 = node_id(ix0 + 1, iy0)
            connect[e] = [N1, N2, N3, N4, N5, N6]; e += 1

            # Second triangle (CCW)
            N1 = node_id(ix0 + 2, iy0 + 2)
            N2 = node_id(ix0,     iy0 + 2)
            N3 = node_id(ix0 + 2, iy0)
            N4 = node_id(ix0 + 1, iy0 + 2)
            N5 = node_id(ix0 + 1, iy0 + 1)
            N6 = node_id(ix0 + 2, iy0 + 1)
            connect[e] = [N1, N2, N3, N4, N5, N6]; e += 1


def generate_tri6_rectangular_mesh_no_error_domain(
    xl: float, yl: float, xh: float, yh: float, nx: int, ny: int
) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUGGY (for testing): silently accept invalid domain extents.
    If xl >= xh or yl >= yh, it swaps the bounds instead of raising.
    This produces a mesh, but violates the spec that requires raising
    ValueError for invalid inputs.
    """
    # --- BUG: silently swap extents instead of raising ---
    if xl >= xh:
        xl, xh = xh, xl
    if yl >= yh:
        yl, yh = yh, yl

    if nx < 1 or ny < 1:
        raise ValueError("nx and ny must be ≥ 1")

    dx = (xh - xl) / float(nx)
    dy = (yh - yl) / float(ny)

    npx, npy = 2 * nx + 1, 2 * ny + 1
    xs = xl + 0.5 * dx * np.arange(npx, dtype=np.float64)
    ys = yl + 0.5 * dy * np.arange(npy, dtype=np.float64)
    XX, YY = np.meshgrid(xs, ys, indexing="xy")
    coords = np.column_stack([XX.ravel(), YY.ravel()])  # (Nnodes, 2) float64

    def node_id(ix: int, iy: int) -> int:
        return iy * npx + ix

    connect = np.empty((2 * nx * ny, 6), dtype=np.int64)
    e = 0
    for cy in range(ny):
        for cx in range(nx):
            ix0 = 2 * cx
            iy0 = 2 * cy

            # First triangle (CCW)
            N1 = node_id(ix0 + 2, iy0)
            N2 = node_id(ix0,     iy0 + 2)
            N3 = node_id(ix0,     iy0)
            N4 = node_id(ix0 + 1, iy0 + 1)
            N5 = node_id(ix0,     iy0 + 1)
            N6 = node_id(ix0 + 1, iy0)
            connect[e] = [N1, N2, N3, N4, N5, N6]; e += 1

            # Second triangle (CCW)
            N1 = node_id(ix0 + 2, iy0 + 2)
            N2 = node_id(ix0,     iy0 + 2)
            N3 = node_id(ix0 + 2, iy0)
            N4 = node_id(ix0 + 1, iy0 + 2)
            N5 = node_id(ix0 + 1, iy0 + 1)
            N6 = node_id(ix0 + 2, iy0 + 1)
            connect[e] = [N1, N2, N3, N4, N5, N6]; e += 1

    return coords, connect


def test_tri6_mesh_basic_structure_and_determinism(fcn):
    """
    Validate basic mesh structure on a 2×2 unit square domain.

    Checks:
    - Correct number of nodes and elements.
    - Array shapes and dtypes are as expected.
    - Corner nodes coincide with the domain boundaries.
    - Node coordinates form a uniform half-step lattice (dx/2, dy/2).
    - Repeated calls yield identical results (determinism).
    """
    xl, yl, xh, yh = 0.0, 0.0, 1.0, 1.0
    nx, ny = 2, 2

    coords1, connect1 = fcn(xl, yl, xh, yh, nx, ny)
    coords2, connect2 = fcn(xl, yl, xh, yh, nx, ny)

    npx, npy = 2*nx + 1, 2*ny + 1
    assert coords1.shape == (npx * npy, 2)
    assert connect1.shape == (2 * nx * ny, 6)
    assert coords1.dtype == np.float64
    assert connect1.dtype in (np.int64, np.int32)

    corner_ids = [0, 2*nx, (2*ny)*npx, (2*ny)*npx + 2*nx]
    expected_corners = np.array([[xl, yl], [xh, yl], [xl, yh], [xh, yh]], dtype=np.float64)
    assert np.allclose(coords1[corner_ids], expected_corners, atol=1e-12)

    dx, dy = (xh - xl)/nx, (yh - yl)/ny
    xs, ys = np.unique(coords1[:, 0]), np.unique(coords1[:, 1])
    assert np.isclose(xs.min(), xl) and np.isclose(xs.max(), xh)
    assert np.isclose(ys.min(), yl) and np.isclose(ys.max(), yh)
    assert np.allclose(np.diff(xs), 0.5*dx)
    assert np.allclose(np.diff(ys), 0.5*dy)

    assert np.array_equal(connect1, connect2)
    assert np.allclose(coords1, coords2)


def test_tri6_mesh_geometry_and_conformity(fcn):
    """
    Validate geometric properties and conformity on a non-square domain.

    Checks:
    - Connectivity indices are within valid range and unique per element.
    - Corner nodes (N1–N3) are consistently counter-clockwise.
    - Midside nodes (N4–N6) equal the average of their adjacent corners.
    - Shared edges between elements reuse identical node IDs (conforming mesh).
    """
    xl, yl, xh, yh = -1.0, 2.0, 3.0, 5.0
    nx, ny = 3, 1
    coords, connect = fcn(xl, yl, xh, yh, nx, ny)

    # --- Index validity and uniqueness ---
    npx, npy = 2*nx + 1, 2*ny + 1
    n_nodes = npx * npy
    assert connect.min() >= 0 and connect.max() < n_nodes
    for e in connect:
        assert len(set(e.tolist())) == 6

    # --- Orientation: corner nodes CCW ---
    for e in connect:
        x1, y1 = coords[e[0]]
        x2, y2 = coords[e[1]]
        x3, y3 = coords[e[2]]
        area2 = (x2 - x1)*(y3 - y1) - (y2 - y1)*(x3 - x1)
        assert area2 > 0.0

    # --- Midside placement ---
    c = coords
    atol = 1e-12
    assert np.allclose(c[connect[:, 3]], 0.5*(c[connect[:, 0]] + c[connect[:, 1]]), atol=atol)
    assert np.allclose(c[connect[:, 4]], 0.5*(c[connect[:, 1]] + c[connect[:, 2]]), atol=atol)
    assert np.allclose(c[connect[:, 5]], 0.5*(c[connect[:, 2]] + c[connect[:, 0]]), atol=atol)

    # --- Conformity: shared edges reuse identical node IDs ---
    edges = []
    for e in connect:
        edges.extend([
            tuple(sorted((e[0], e[1]))),
            tuple(sorted((e[1], e[2]))),
            tuple(sorted((e[2], e[0])))
        ])

    # Each edge must occur at most twice (boundary=1, interior=2)
    for edge in set(edges):
        count = edges.count(edge)
        assert count in (1, 2)
    # And at least one interior edge exists in this mesh
    assert any(edges.count(edge) == 2 for edge in set(edges))


def test_tri6_mesh_invalid_inputs(fcn):
    """
    Validate error handling for invalid inputs.

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
    task_id = "generate_tri6_rectangular_mesh"
    task_short_description = "generates a rectangular mesh of tri6 elements"
    created_date = "2025-09-22"
    created_by = "elejeune11"
    main_fcn = generate_tri6_rectangular_mesh
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
    test_cases = [{"test_code": test_tri6_mesh_basic_structure_and_determinism, "expected_failures": [generate_tri6_rectangular_mesh_indexing_bug]},
                  {"test_code": test_tri6_mesh_geometry_and_conformity, "expected_failures": [generate_tri6_rectangular_mesh_ccw_bug, generate_tri6_rectangular_mesh_duplicate_node_bug]},
                  {"test_code": test_tri6_mesh_invalid_inputs, "expected_failures": [generate_tri6_rectangular_mesh_no_error_nx_ny, generate_tri6_rectangular_mesh_no_error_domain]}
                  ]
    return task_id, task_short_description, created_date, created_by, main_fcn, required_imports, fcn_dependencies, reference_verification_inputs, test_cases
