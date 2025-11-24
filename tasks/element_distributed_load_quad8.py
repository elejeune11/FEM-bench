import numpy as np
import pytest
from typing import Tuple


def element_distributed_load_quad8(
    face: int,
    node_coords: np.ndarray,
    traction: np.ndarray,
    num_gauss_pts: int,
) -> np.ndarray:
    """
    Assemble the consistent nodal load vector for a single edge of a Q8
    (8-node quadratic quadrilateral) under a constant traction and
    return it scattered into the full element DOF vector of length 16:
    [Fx1, Fy1, Fx2, Fy2, …, Fx8, Fy8].

    Traction model
    --------------
    `traction` is a constant Cauchy traction (force per unit physical
    edge length) applied along the current edge:
        traction = [t_x, t_y]  (shape (2,))

    Expected Q8 node ordering (must match `node_coords`)
    ----------------------------------------------------
        1:(-1,-1), 2:( 1,-1), 3:( 1, 1), 4:(-1, 1),
        5:( 0,-1), 6:( 1, 0), 7:( 0, 1), 8:(-1, 0)

    Face orientation & edge connectivity (start, mid, end)
    ------------------------------------------------------
        face=0 (bottom): (0, 4, 1)
        face=1 (right) : (1, 5, 2)
        face=2 (top)   : (2, 6, 3)
        face=3 (left)  : (3, 7, 0)
    The local edge parameter s runs from the start corner (s=-1) to the
    end corner (s=+1), with the mid-edge node at s=0.

    Parameters
    ----------
    face : {0,1,2,3}
        Which edge to load (bottom, right, top, left in the reference element).
    node_coords : (8,2) float array
        Physical coordinates of the Q8 nodes in the expected ordering above.
    traction : (2,) float array
        Constant Cauchy traction vector [t_x, t_y].
    num_gauss_pts : {1,2,3}
        1D Gauss–Legendre points on [-1,1]. (For straight edges,
        2-pt is exact with constant traction.)

    Returns
    -------
    r_elem : (16,) float array
        Element load vector in DOF order [Fx1, Fy1, Fx2, Fy2, …, Fx8, Fy8].
        Only the three edge nodes receive nonzero entries; others are zero.
    """
    # ---- Face connectivity (start corner, mid-edge, end corner)
    face_nodes = {
        0: (0, 4, 1),  # bottom
        1: (1, 5, 2),  # right
        2: (2, 6, 3),  # top
        3: (3, 7, 0),  # left
    }[int(face)]

    # ---- Slice the 3 edge nodes (x,y) and constant traction
    edge_xy = np.asarray(node_coords, dtype=float)[list(face_nodes), :]  # (3,2)
    t = np.asarray(traction, dtype=float).reshape(2)                      # (2,)
    tx, ty = float(t[0]), float(t[1])

    # ---- Gauss–Legendre on [-1,1]
    def gauss_1d(n: int) -> Tuple[np.ndarray, np.ndarray]:
        if n == 1:
            return np.array([0.0]), np.array([2.0])
        if n == 2:
            a = 1.0 / np.sqrt(3.0)
            return np.array([-a, a]), np.array([1.0, 1.0])
        if n == 3:
            b = np.sqrt(3.0 / 5.0)
            return np.array([-b, 0.0, b]), np.array([5.0/9.0, 8.0/9.0, 5.0/9.0])
        # fallback to 2-pt
        a = 1.0 / np.sqrt(3.0)
        return np.array([-a, a]), np.array([1.0, 1.0])

    # ---- 1D quadratic line shape functions and derivatives (s ∈ [-1,1])
    def N_line(s: float) -> np.ndarray:
        s2 = s * s
        return np.array([0.5 * (s2 - s), 1.0 - s2, 0.5 * (s2 + s)], dtype=float)  # [start, mid, end]

    def dNds_line(s: float) -> np.ndarray:
        return np.array([s - 0.5, -2.0 * s, s + 0.5], dtype=float)

    # ---- Integrate along the chosen edge (edge-local accumulation)
    s_pts, w_pts = gauss_1d(int(num_gauss_pts))
    r_edge = np.zeros(6, dtype=float)  # [Fx(start),Fy(start), Fx(mid),Fy(mid), Fx(end),Fy(end)]

    for s, w in zip(s_pts, w_pts):
        s = float(s); w = float(w)
        N    = N_line(s)
        dNds = dNds_line(s)

        # Physical line measure |∂x/∂s|
        dxds = float(edge_xy[:, 0] @ dNds)
        dyds = float(edge_xy[:, 1] @ dNds)
        meas = np.hypot(dxds, dyds)

        scale = w * meas
        # start
        r_edge[0] += scale * N[0] * tx
        r_edge[1] += scale * N[0] * ty
        # mid
        r_edge[2] += scale * N[1] * tx
        r_edge[3] += scale * N[1] * ty
        # end
        r_edge[4] += scale * N[2] * tx
        r_edge[5] += scale * N[2] * ty

    # ---- Scatter edge-local (start,mid,end) into full element vector
    r_elem = np.zeros(16, dtype=float)  # [Fx1,Fy1, Fx2,Fy2, …, Fx8,Fy8]
    for k, node_idx in enumerate(face_nodes):  # k=0,1,2
        base = 2 * node_idx
        r_elem[base]     += r_edge[2*k + 0]  # Fx(node)
        r_elem[base + 1] += r_edge[2*k + 1]  # Fy(node

    return r_elem


def element_distributed_load_quad8__all_ones(face, node_coords, traction, num_gauss_pts=2):
    """
    Expected-failure variant of element_distributed_load_quad8.

    Instead of computing anything, it just returns an element load vector
    of all ones. This guarantees the same output shape (16,) as the real
    function but will fail analytical/resultant checks.

    Returns
    -------
    r_elem : (16,) float array
        All entries = 1.0.
    """
    return np.ones(16, dtype=float)


def test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces(fcn):
    """
    Test that the traction integral works on straigt edge elements.
    Set up straight edges on an 8 node quadrilateral element uniformly scaled by 2x.
    For each face (0=bottom, 1=right, 2=top, 3=left), apply a constant Cauchy
    traction t = [t_x, t_y].
    Check two things:
    1. The total force recovered from summing nodal contributions along the
    loaded edge matches the applied traction times the physical edge length.
    2. All nodes that are not on the loaded edge have zero load.
    """
    # Reference Q8 → scale by 2× about origin
    NC_ref = np.array([
        [-1.0, -1.0],  # 0
        [ 1.0, -1.0],  # 1
        [ 1.0,  1.0],  # 2
        [-1.0,  1.0],  # 3
        [ 0.0, -1.0],  # 4
        [ 1.0,  0.0],  # 5
        [ 0.0,  1.0],  # 6
        [-1.0,  0.0],  # 7
    ], dtype=float)
    NC = 2.0 * NC_ref  # all edges straight; each has physical length L = 4

    # Constant traction (shape (2,))
    tx, ty = 2.5, -1.3
    traction = np.array([tx, ty], dtype=float)

    # Face connectivity (start, mid, end) for Q8
    face_nodes_map = {
        0: (0, 4, 1),  # bottom
        1: (1, 5, 2),  # right
        2: (2, 6, 3),  # top
        3: (3, 7, 0),  # left
    }

    L_edge = 4.0
    tol = 1e-13

    for face in (0, 1, 2, 3):
        r = fcn(
            face=face, node_coords=NC, traction=traction, num_gauss_pts=2
        )
        assert isinstance(r, np.ndarray)
        assert r.shape == (16,)

        # Resultant over ALL nodes: non-face DOFs should be zero, but summing
        # all makes that implicit and also checks scattering is correct.
        Fx_sum = np.sum(r[0::2])
        Fy_sum = np.sum(r[1::2])
        assert np.allclose(Fx_sum, tx * L_edge, rtol=tol, atol=tol)
        assert np.allclose(Fy_sum, ty * L_edge, rtol=tol, atol=tol)

        # Additionally, assert non-face node DOFs are ~0.
        face_nodes = set(face_nodes_map[face])
        for node in range(8):
            fx_idx = 2 * node
            fy_idx = fx_idx + 1
            if node in face_nodes:
                # Edge nodes should carry (possibly nonzero) loads — no assertion here
                continue
            # Non-face nodes must be zero
            assert abs(r[fx_idx]) <= tol
            assert abs(r[fy_idx]) <= tol


def test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge(fcn):
    """
    Test the performance of curved edges.
    Curved bottom edge (face=0) parameterized by s ∈ [-1, 1]:
        x(s) = s,  y(s) = c + k s^2  (parabola through the three edge nodes)
    realized by placing 8 node quadrilateral edge nodes as:
        start = (-1, c+k),  mid = (0, c),  end = (1, c+k).

    With a constant Cauchy traction t = [t_x, t_y], check that the total force equals
        [t_x, t_y] * L_exact,
    where the exact arc length on [-1,1] is
        L_exact = sqrt(1+α) + asinh(sqrt(α)) / sqrt(α),   α = 4 k^2.

    Note that the function integrates with 3-point Gauss–Legendre along the curved edge.
    The integrand involves sqrt(1+α s^2), which is not a polynomial, so the
    3-point rule is not exact. Select an appropriate relative tolerance to address this.
    """
    # Parabola parameters (k ≠ 0 to ensure curvature)
    c, k = -1.0, 0.35
    alpha = 4.0 * k * k

    # Build Q8 node coordinates (only nodes 0,4,1 affect face=0)
    NC = np.array([
        [-1.0, c + k],  # 0 start
        [ 1.0, c + k],  # 1 end
        [ 1.0,  1.0],   # 2
        [-1.0,  1.0],   # 3
        [ 0.0,  c    ], # 4 mid
        [ 1.0,  0.0],   # 5
        [ 0.0,  1.0],   # 6
        [-1.0,  0.0],   # 7
    ], dtype=float)

    # Constant traction (shape (2,))
    tx, ty = 1.7, -0.9
    traction = np.array([tx, ty], dtype=float)

    # 3-pt Gauss: accurate for smooth curved edges (not exact for sqrt)
    r = fcn(face=0, node_coords=NC, traction=traction, num_gauss_pts=3)
    assert isinstance(r, np.ndarray)
    assert r.shape == (16,)

    # FE resultants: sum over ALL nodes (non-face DOFs should be zero)
    Fx_sum = np.sum(r[0::2])
    Fy_sum = np.sum(r[1::2])

    # Exact arc length
    if alpha == 0.0:
        L_exact = 2.0
    else:
        L_exact = np.sqrt(1.0 + alpha) + np.arcsinh(np.sqrt(alpha)) / np.sqrt(alpha)

    expected_Fx = tx * L_exact
    expected_Fy = ty * L_exact

    # Allow a small relative tolerance due to non-polynomial integrand with 3-pt GL
    assert np.allclose(Fx_sum, expected_Fx, rtol=5e-4, atol=1e-12)
    assert np.allclose(Fy_sum, expected_Fy, rtol=5e-4, atol=1e-12)


def task_info():
    task_id = "element_distributed_load_quad8"
    task_short_description = "compute the equivalent nodal load for distributed load applied to the face of 8 node quad elements"
    created_date = "2025-09-24"
    created_by = "elejeune11"
    main_fcn = element_distributed_load_quad8
    required_imports = ["import numpy as np", "import pytest", "from typing import Tuple"]
    fcn_dependencies = []
    reference_verification_inputs = [
        # 1) Straight bottom edge, constant traction
        [
            0,
            np.array([  # reference Q8
                [-1.0, -1.0],  # 0
                [ 1.0, -1.0],  # 1
                [ 1.0,  1.0],  # 2
                [-1.0,  1.0],  # 3
                [ 0.0, -1.0],  # 4
                [ 1.0,  0.0],  # 5
                [ 0.0,  1.0],  # 6
                [-1.0,  0.0],  # 7
            ], dtype=float),
            np.array([2.0, 1.0], dtype=float),  # t = [tx, ty] (constant)
            1,
        ],
        # 2) Right edge, constant traction
        [
            1,
            np.array([
                [-1.0, -1.0],
                [ 1.0, -1.0],
                [ 1.0,  1.0],
                [-1.0,  1.0],
                [ 0.0, -1.0],
                [ 1.0,  0.0],
                [ 0.0,  1.0],
                [-1.0,  0.0],
            ], dtype=float),
            np.array([0.3, -0.5], dtype=float),
            2,
        ],
        # 3) Top edge, constant traction, higher quadrature
        [
            2,
            np.array([
                [-1.0, -1.0],
                [ 1.0, -1.0],
                [ 1.0,  1.0],
                [-1.0,  1.0],
                [ 0.0, -1.0],
                [ 1.0,  0.0],
                [ 0.0,  1.0],
                [-1.0,  0.0],
            ], dtype=float),
            np.array([1.0, -1.0], dtype=float),
            3,
        ],
        # 4) Left edge, scaled geometry (2× larger element), constant traction
        [
            3,
            2.0 * np.array([
                [-1.0, -1.0],
                [ 1.0, -1.0],
                [ 1.0,  1.0],
                [-1.0,  1.0],
                [ 0.0, -1.0],
                [ 1.0,  0.0],
                [ 0.0,  1.0],
                [-1.0,  0.0],
            ], dtype=float),
            np.array([0.0, 3.0], dtype=float),
            2,
        ],
        # 5) Bottom edge, curved geometry (mid-edge lifted), constant traction
        [
            0,
            np.array([
                [-1.0, -1.0],   # start
                [ 1.0, -1.0],   # end
                [ 1.0,  1.0],
                [-1.0,  1.0],
                [ 0.0, -0.5],   # mid node lifted upward (curved edge)
                [ 1.0,  0.0],
                [ 0.0,  1.0],
                [-1.0,  0.0],
            ], dtype=float),
            np.array([2.0, 1.0], dtype=float),
            3,
        ],
    # 6) Right edge, curved geometry (mid-edge bulged left), constant traction
        [
            1,
            np.array([
                [-1.0, -1.0],  # 0
                [ 1.0, -1.0],  # 1 (start of right edge)
                [ 1.0,  1.0],  # 2 (end   of right edge)
                [-1.0,  1.0],  # 3
                [ 0.0, -1.0],  # 4
                [ 0.85, 0.0],  # 5 mid-edge shifted left → curved right edge
                [ 0.0,  1.0],  # 6
                [-1.0,  0.0],  # 7
            ], dtype=float),
            np.array([1.2, 0.4], dtype=float),  # t = [tx, ty]
            3,
        ],
        # 7) Top edge, curved geometry (mid-edge bulged upward with slight x-shift), constant traction
        [
            2,
            np.array([
                [-1.0, -1.0],  # 0
                [ 1.0, -1.0],  # 1
                [ 1.0,  1.0],  # 2 (start of top edge)
                [-1.0,  1.0],  # 3 (end   of top edge)
                [ 0.0, -1.0],  # 4
                [ 1.0,  0.0],  # 5
                [ 0.10, 1.15], # 6 mid-edge shifted up/right → curved top edge
                [-1.0,  0.0],  # 7
            ], dtype=float),
            np.array([-0.7, 0.9], dtype=float),
            3,
        ],
        # 8) Left edge, curved geometry (mid-edge bulged right/down), constant traction
        [
            3,
            np.array([
                [-1.0, -1.0],  # 0 (end of left edge)
                [ 1.0, -1.0],  # 1
                [ 1.0,  1.0],  # 2
                [-1.0,  1.0],  # 3 (start of left edge)
                [ 0.0, -1.0],  # 4
                [ 1.0,  0.0],  # 5
                [ 0.0,  1.0],  # 6
                [-0.90, -0.10],# 7 mid-edge shifted right/down → curved left edge
            ], dtype=float),
            np.array([0.5, -1.4], dtype=float),
            3,
        ],
    ]
    test_cases = [{"test_code": test_edl_q8_analytic_straight_edges_total_force_scaled_all_faces, "expected_failures": [element_distributed_load_quad8__all_ones]},
                  {"test_code": test_edl_q8_constant_traction_total_force_on_curved_parabolic_edge, "expected_failures": [element_distributed_load_quad8__all_ones]}
                  ]
    return {
        "task_id": task_id,
        "task_short_description": task_short_description,
        "created_date": created_date,
        "created_by": created_by,
        "main_fcn": main_fcn,
        "required_imports": required_imports,
        "fcn_dependencies": fcn_dependencies,
        "reference_verification_inputs": reference_verification_inputs,
        "test_cases": test_cases,
        # "python_version": "version_number",
        # "package_versions": {"numpy": "version_number", },
    }
