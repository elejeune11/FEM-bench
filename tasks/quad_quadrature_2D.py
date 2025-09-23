import numpy as np
import pytest
from typing import Tuple


def quad_quadrature_2D(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return Gauss–Legendre quadrature points and weights for the
    reference square Q = { (xi, eta) : -1 <= xi <= 1, -1 <= eta <= 1 }.

    Supported rules (tensor products of 1D Gauss–Legendre):
      - 1 point  : 1×1   (exact for polynomials with degree ≤1 in each variable)
      - 4 points : 2×2   (exact for degree ≤3 in each variable)
      - 9 points : 3×3   (exact for degree ≤5 in each variable)

    Parameters
    ----------
    num_pts : int
        Total number of quadrature points (1, 4, or 9).

    Returns
    -------
    points : (num_pts, 2) float64 ndarray
        Quadrature points [xi, eta] on the reference square.
    weights : (num_pts,) float64 ndarray
        Quadrature weights corresponding to `points`. The sum of weights
        equals the area of Q, which is 4.0.

    Raises
    ------
    ValueError
        If `num_pts` is not one of {1, 4, 9}.
    """
    if num_pts not in (1, 4, 9):
        raise ValueError("num_pts must be one of {1, 4, 9}.")

    # Select 1D rule
    if num_pts == 1:
        nodes_1d = np.array([0.0], dtype=np.float64)
        w_1d = np.array([2.0], dtype=np.float64)
    elif num_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        nodes_1d = np.array([-a, +a], dtype=np.float64)
        w_1d = np.array([1.0, 1.0], dtype=np.float64)
    else:  # num_pts == 9
        b = np.sqrt(3.0 / 5.0)
        nodes_1d = np.array([-b, 0.0, +b], dtype=np.float64)
        w_1d = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0], dtype=np.float64)

    # Tensor product (xi varies fastest)
    XI, ETA = np.meshgrid(nodes_1d, nodes_1d, indexing="xy")
    points = np.column_stack([XI.ravel(order="C"), ETA.ravel(order="C")]).astype(np.float64, copy=False)

    WXI, WETA = np.meshgrid(w_1d, w_1d, indexing="xy")
    weights = (WXI * WETA).ravel(order="C").astype(np.float64, copy=False)

    return points, weights


def quad_quadrature_2D_expected_failure_no_error(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUGGY version (for testing expected failure).

    Incorrectly accepts unsupported numbers of quadrature points.
    Instead of raising ValueError for invalid inputs, it silently
    returns empty arrays. This violates the specification and will
    cause the invalid-input test to fail.
    """
    if num_pts == 1:
        points = np.array([[0.0, 0.0]], dtype=np.float64)
        weights = np.array([4.0], dtype=np.float64)
    elif num_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        points = np.array([[-a, -a], [a, -a], [-a, a], [a, a]], dtype=np.float64)
        weights = np.ones(4, dtype=np.float64)
    elif num_pts == 9:
        b = np.sqrt(3.0 / 5.0)
        nodes = np.array([-b, 0.0, b], dtype=np.float64)
        w1d = np.array([5.0/9.0, 8.0/9.0, 5.0/9.0], dtype=np.float64)
        XI, ETA = np.meshgrid(nodes, nodes, indexing="xy")
        points = np.column_stack([XI.ravel(), ETA.ravel()]).astype(np.float64, copy=False)
        WXI, WETA = np.meshgrid(w1d, w1d, indexing="xy")
        weights = (WXI * WETA).ravel().astype(np.float64, copy=False)
    else:
        # BUG: should raise ValueError, but instead silently returns empty arrays
        points = np.empty((0, 2), dtype=np.float64)
        weights = np.empty((0,), dtype=np.float64)
    return points, weights


def quad_quadrature_2D_expected_failure_basics(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUGGY version (for testing expected failure).

    Incorrectly normalizes weights so they sum to 1.0 instead of 4.0
    (the area of the reference square). This will fail the basics test.
    """
    if num_pts == 1:
        points = np.array([[0.0, 0.0]], dtype=np.float64)
        weights = np.array([1.0], dtype=np.float64)  # BUG: should be 4.0
    elif num_pts == 4:
        a = 1.0 / np.sqrt(3.0)
        points = np.array([[-a, -a], [a, -a], [-a, a], [a, a]], dtype=np.float64)
        weights = np.full(4, 0.25, dtype=np.float64)  # BUG: sums to 1.0
    elif num_pts == 9:
        b = np.sqrt(3.0 / 5.0)
        nodes = np.array([-b, 0.0, b], dtype=np.float64)
        XI, ETA = np.meshgrid(nodes, nodes, indexing="xy")
        points = np.column_stack([XI.ravel(), ETA.ravel()]).astype(np.float64, copy=False)
        weights = np.full(9, 1.0/9.0, dtype=np.float64)  # BUG: sums to 1.0
    else:
        raise ValueError("num_pts must be 1, 4, or 9")
    return points, weights


def quad_quadrature_2D_expected_failure_zeros(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUGGY version (for expected failure).

    Always returns zero points and zero weights. This fails because even the
    integral of 1 over the square should be 4.0, but is approximated as 0.0.
    """
    points = np.zeros((num_pts, 2), dtype=np.float64)
    weights = np.zeros((num_pts,), dtype=np.float64)
    return points, weights


def quad_quadrature_2D_expected_failure_ones(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUGGY version (for expected failure).

    Always returns points and weights filled with ones, regardless of num_pts.
    This ensures degree-exactness and basic checks (area, domain) will fail.
    """
    points = np.ones((num_pts, 2), dtype=np.float64)
    weights = np.ones((num_pts,), dtype=np.float64)
    return points, weights


def test_quad_quadrature_2D_invalid_inputs(fcn):
    """
    Test that quad_quadrature_2D rejects invalid numbers of points.

    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fcn(0)
    with pytest.raises(ValueError):
        fcn(2)
    with pytest.raises(ValueError):
        fcn(3)
    with pytest.raises(ValueError):
        fcn(5)
    with pytest.raises(ValueError):
        fcn(7)


def test_quad_quadrature_2D_basics(fcn):
    """
    Test basic structural properties of the quadrature rule for quads.

    For each supported rule (1, 4, 9 points):
    - The returned points and weights arrays have the correct shapes and dtypes.
    - The weights sum to 4.0, which is the exact area of the reference square [-1,1] x [-1,1].
    - All quadrature points lie inside the reference square, i.e.
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for n in (1, 4, 9):
        pts, w = fcn(n)

        assert pts.shape == (n, 2)
        assert w.shape == (n,)
        assert pts.dtype == np.float64
        assert w.dtype == np.float64

        # weights must sum to the area of the square = 4
        assert np.isclose(w.sum(), 4.0)

        x, y = pts[:, 0], pts[:, 1]
        assert np.all(x >= -1.0 - 1e-14)
        assert np.all(x <=  1.0 + 1e-14)
        assert np.all(y >= -1.0 - 1e-14)
        assert np.all(y <=  1.0 + 1e-14)


def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """
    Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
    reference square [-1,1]×[-1,1].

    Background
    ----------
    • The tensor-product 1-point rule places a single node at the center (0,0) with
      weight 4. This integrates exactly any polynomial that is at most degree 1
      in each variable, i.e. constants and linear terms in x or y.
    • For higher-degree terms, the rule is no longer guaranteed to be exact.

    What this test does
    -------------------
    1. Constructs a random affine polynomial P(x,y) = a00 + a10·x + a01·y.
       - Checks that the quadrature result matches the analytic integral.
    2. Adds quadratic contributions (x², y², xy).
       - Demonstrates that the quadrature result deviates from the analytic value,
         confirming that the 1-point rule is not exact for degree ≥ 2.

    Expected outcome
    ----------------
    • Exactness assertions for monomials of degree ≤ 1 should pass.
    • Non-exactness assertions for quadratics should fail the exactness check
      (i.e. the quadrature does not reproduce the analytic integrals).
    """
    rng = np.random.default_rng(123)
    pts, w = fcn(1)
    x, y = pts[:, 0], pts[:, 1]

    # Exact case: P(x,y) = a00 + a10 x + a01 y
    a00, a10, a01 = rng.standard_normal(3)
    f_exact = a00 + a10 * x + a01 * y
    approx = np.dot(w, f_exact)

    # Exact integral via separability: ∑ a_pq I(p) I(q)
    def I(n): return 0.0 if (n % 2) else 2.0 / (n + 1)
    exact = a00 * I(0) * I(0) + a10 * I(1) * I(0) + a01 * I(0) * I(1)
    assert np.isclose(approx, exact, rtol=1e-13, atol=1e-15)

    # Non-exact: include quadratic terms (deg_x=2 or deg_y=2)
    b20, b02, b11 = rng.standard_normal(3)
    f_fail = f_exact + b20 * x**2 + b02 * y**2 + b11 * x * y
    approx_fail = np.dot(w, f_fail)
    exact_fail = exact + b20 * I(2) * I(0) + b02 * I(0) * I(2) + b11 * I(1) * I(1)
    assert not np.isclose(approx_fail, exact_fail, rtol=1e-12, atol=1e-14)


def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """
    Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].

    Background
    ----------
    • The 2-point Gauss–Legendre rule in 1D integrates exactly any polynomial up to degree 3.
      Taking the tensor product yields a 2×2 rule in 2D (4 points total), which is exact for
      any polynomial with degree ≤ 3 in each variable separately (i.e., cubic polynomials
      in x and y).
    • It is not guaranteed to integrate quartic terms (degree 4 in either variable) exactly.

    What this test does
    -------------------
    1. Builds a random polynomial P(x,y) = Σ a_ij x^i y^j for 0 ≤ i,j ≤ 3, i.e. all terms up
       to cubic in x and cubic in y. The quadrature approximation is compared to the analytic
       integral, the two results must match.
    2. Extends the polynomial with quartic terms x^4 and y^4. For these terms the 2×2 rule
       is not exact, so the quadrature result is expected to deviate from the analytic integral.

    Expected outcome
    ----------------
    • Exactness assertions for all monomials with per-variable degree ≤ 3 should pass.
    • Adding quartic terms should break exactness, and the mismatch is detected by the test.
    """
    rng = np.random.default_rng(456)
    pts, w = fcn(4)
    x, y = pts[:, 0], pts[:, 1]

    def I(n): return 0.0 if (n % 2) else 2.0 / (n + 1)

    # Build random polynomial sum_{i=0..3} sum_{j=0..3} a_ij x^i y^j
    A = rng.standard_normal((4, 4))
    f = np.zeros_like(w)
    exact = 0.0
    for i in range(4):
        for j in range(4):
            term = A[i, j] * (x**i) * (y**j)
            f += term
            exact += A[i, j] * I(i) * I(j)

    approx = np.dot(w, f)
    assert np.isclose(approx, exact, rtol=1e-12, atol=1e-14)

    # Add a single degree-4 term to break exactness
    c40, c04 = rng.standard_normal(2)
    f_break = f + c40 * x**4 + c04 * y**4
    exact_break = exact + c40 * I(4) * I(0) + c04 * I(0) * I(4)
    approx_break = np.dot(w, f_break)
    assert not np.isclose(approx_break, exact_break, rtol=1e-12, atol=1e-14)


def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """
    Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].

    Background
    ----------
    • The 3-point Gauss–Legendre rule in 1D integrates polynomials exactly up to degree 5.
      Taking the tensor product yields a 3×3 rule in 2D (9 points total), which is exact
      for any polynomial where the degree in each variable is ≤ 5.
    • The rule is not guaranteed to integrate terms with degree 6 or higher in either variable.

    What this test does
    -------------------
    1. Constructs a random polynomial P(x,y) = Σ a_ij x^i y^j with 0 ≤ i,j ≤ 5 (up to quintic
       terms in both variables). The quadrature approximation is compared to the analytic
       integral, the two results must match within tight tolerance.
    2. Extends the polynomial with degree-6 contributions (x^6, y^6, x^4y^2). These exceed the
       rule’s guaranteed exactness, so the quadrature result is expected to deviate from the
       analytic integral.

    Expected outcome
    ----------------
    • Exactness assertions for all monomials with per-variable degree ≤ 5 should pass.
    • Adding degree-6 terms should break exactness, and the mismatch is detected by the test.
    """
    rng = np.random.default_rng(789)
    pts, w = fcn(9)
    x, y = pts[:, 0], pts[:, 1]

    def I(n): return 0.0 if (n % 2) else 2.0 / (n + 1)

    # Random polynomial up to degree 5 in each variable
    A = rng.standard_normal((6, 6))  # indices 0..5
    f = np.zeros_like(w)
    exact = 0.0
    for i in range(6):
        for j in range(6):
            term = A[i, j] * (x**i) * (y**j)
            f += term
            exact += A[i, j] * I(i) * I(j)

    approx = np.dot(w, f)
    assert np.isclose(approx, exact, rtol=1e-12, atol=1e-14)

    # Add degree-6 terms to show non-exactness (avoid odd*odd like (3,3))
    c60, c06, c42 = rng.standard_normal(3)
    f_break = f + c60 * x**6 + c06 * y**6 + c42 * (x**4) * (y**2)
    exact_break = exact + c60 * I(6) * I(0) + c06 * I(0) * I(6) + c42 * I(4) * I(2)
    approx_break = np.dot(w, f_break)
    assert not np.isclose(approx_break, exact_break, rtol=1e-12, atol=1e-14)


def task_info():
    task_id = "quad_quadrature_2D"
    task_short_description = "returns quadrature points and weights for 2d quad elements"
    created_date = "2025-09-22"
    created_by = "elejeune11"
    main_fcn = quad_quadrature_2D
    required_imports = ["import numpy as np", "import pytest", "from typing import Tuple"]
    fcn_dependencies = []
    reference_verification_inputs = [[1], [4], [9]]
    test_cases = [{"test_code": test_quad_quadrature_2D_invalid_inputs, "expected_failures": [quad_quadrature_2D_expected_failure_no_error]},
                  {"test_code": test_quad_quadrature_2D_basics, "expected_failures": [quad_quadrature_2D_expected_failure_basics]},
                  {"test_code": test_quad_quadrature_2D_degree_exactness_1pt, "expected_failures": [quad_quadrature_2D_expected_failure_zeros, quad_quadrature_2D_expected_failure_ones]},
                  {"test_code": test_quad_quadrature_2D_degree_exactness_2x2, "expected_failures": [quad_quadrature_2D_expected_failure_zeros, quad_quadrature_2D_expected_failure_ones]},
                  {"test_code": test_quad_quadrature_2D_degree_exactness_3x3, "expected_failures": [quad_quadrature_2D_expected_failure_zeros, quad_quadrature_2D_expected_failure_ones]}
                  ]
    return task_id, task_short_description, created_date, created_by, main_fcn, required_imports, fcn_dependencies, reference_verification_inputs, test_cases
