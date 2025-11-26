import math
import numpy as np
import pytest
from typing import Tuple


def triangle_quadrature_2D(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Return quadrature points and weights for numerical integration
    over the reference triangle T = {(x,y): x>=0, y>=0, x+y<=1}.

    Supported rules:
    - 1-point (degree-1 exact): centroid.
    - 3-point (degree-2 exact): permutations of (1/6, 1/6).
    - 4-point (degree-3 exact): centroid + permutations of (0.6, 0.2).

    Parameters
    ----------
    num_pts : int
        Number of quadrature points (1, 3, or 4).

    Returns
    -------
    points : (num_pts, 2) ndarray of float64
        Quadrature points (xi, eta). The third barycentric coordinate is 1 - xi - eta.
    weights : (num_pts,) ndarray of float64
        Quadrature weights. The sum of weights equals the area of the reference triangle (1/2).

    Raises
    ------
    ValueError
        If `num_pts` is not 1, 3, or 4.
    """
    if num_pts == 1:
        points = np.array([[1.0/3.0, 1.0/3.0]], dtype=np.float64)
        weights = np.array([0.5], dtype=np.float64)

    elif num_pts == 3:
        points = np.array([
            [1.0/6.0, 1.0/6.0],
            [2.0/3.0, 1.0/6.0],
            [1.0/6.0, 2.0/3.0],
        ], dtype=np.float64)
        weights = np.array([1.0/6.0, 1.0/6.0, 1.0/6.0], dtype=np.float64)

    elif num_pts == 4:
        points = np.array([
            [1.0/3.0, 1.0/3.0],
            [0.6,     0.2    ],
            [0.2,     0.6    ],
            [0.2,     0.2    ],
        ], dtype=np.float64)
        weights = np.array([-27.0/96.0, 25.0/96.0, 25.0/96.0, 25.0/96.0], dtype=np.float64)

    else:
        raise ValueError("num_pts must be one of {1, 3, 4}.")

    return points, weights


def triangle_quadrature_2D_expected_failure_no_error(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUGGY version (for testing expected failure).

    Incorrectly accepts unsupported numbers of quadrature points.
    Instead of raising ValueError for invalid inputs, it silently
    returns empty arrays. This violates the specification and will
    cause the invalid-input test to fail.
    """
    if num_pts == 1:
        points = np.array([[1.0/3.0, 1.0/3.0]], dtype=np.float64)
        weights = np.array([0.5], dtype=np.float64)

    elif num_pts == 3:
        points = np.array([
            [1.0/6.0, 1.0/6.0],
            [2.0/3.0, 1.0/6.0],
            [1.0/6.0, 2.0/3.0],
        ], dtype=np.float64)
        weights = np.array([1/6, 1/6, 1/6], dtype=np.float64)

    elif num_pts == 4:
        points = np.array([
            [1/3, 1/3],
            [0.6, 0.2],
            [0.2, 0.6],
            [0.2, 0.2],
        ], dtype=np.float64)
        weights = np.array([-27/96, 25/96, 25/96, 25/96], dtype=np.float64)

    else:
        # BUG: should raise ValueError, but instead silently returns empty arrays
        points = np.empty((0, 2), dtype=np.float64)
        weights = np.empty((0,), dtype=np.float64)

    return points, weights


def triangle_quadrature_2D_expected_failure_basics(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUGGY version (for testing expected failure).

    This function incorrectly normalizes the weights so that they sum to 1.0
    instead of 0.5 (the true area of the reference triangle). This will cause
    test_triangle_quadrature_2D_basics to fail.
    """
    if num_pts == 1:
        points = np.array([[1/3, 1/3]], dtype=np.float64)
        weights = np.array([1.0], dtype=np.float64)  # BUG: should be 0.5

    elif num_pts == 3:
        points = np.array([
            [1/6, 1/6],
            [2/3, 1/6],
            [1/6, 2/3],
        ], dtype=np.float64)
        weights = np.array([1/3, 1/3, 1/3], dtype=np.float64)  # BUG: sums to 1.0

    elif num_pts == 4:
        points = np.array([
            [1/3, 1/3],
            [0.6, 0.2],
            [0.2, 0.6],
            [0.2, 0.2],
        ], dtype=np.float64)
        weights = np.array([-27/48, 25/48, 25/48, 25/48], dtype=np.float64)  # BUG: sum = 1.0

    else:
        raise ValueError("num_pts must be 1, 3, or 4")

    return points, weights


def triangle_quadrature_2D_expected_failure_zeros(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUGGY version (for expected failure).

    Always returns zero points and zero weights.
    This fails because even the integral of 1 over the triangle
    will be approximated as 0 instead of 0.5.
    """
    points = np.zeros((num_pts, 2), dtype=np.float64)
    weights = np.zeros((num_pts,), dtype=np.float64)
    return points, weights


def triangle_quadrature_2D_expected_failure_ones(num_pts: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    BUGGY version (for expected failure).

    Always returns points and weights filled with ones, regardless of num_pts.
    This ensures the degree-exactness tests will fail since the quadrature rule
    is not consistent with any correct integration scheme.
    """
    points = np.ones((num_pts, 2), dtype=np.float64)
    weights = np.ones((num_pts,), dtype=np.float64)
    return points, weights


def test_triangle_quadrature_2D_invalid_inputs(fcn):
    """
    Test that triangle_quadrature_2D rejects invalid numbers of points.

    The quadrature rule only supports 1, 3, or 4 integration points.
    Any other request should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fcn(0)
    with pytest.raises(ValueError):
        fcn(2)
    with pytest.raises(ValueError):
        fcn(5)


def test_triangle_quadrature_2D_basics(fcn):
    """
    Test basic structural properties of the quadrature rule.

    For each supported rule (1, 3, 4 points):
    - The returned points and weights arrays have the correct shapes and dtypes.
    - The weights sum to 1/2, which is the exact area of the reference triangle.
    - All quadrature points lie inside the reference triangle, i.e.
      x >= 0, y >= 0, and x + y <= 1.
    """
    for n in (1, 3, 4):
        pts, w = fcn(n)

        assert pts.shape == (n, 2)
        assert w.shape == (n,)
        assert pts.dtype == np.float64
        assert w.dtype == np.float64

        assert np.isclose(w.sum(), 0.5)

        x, y = pts[:, 0], pts[:, 1]
        s = x + y
        assert np.all(x >= -1e-14)
        assert np.all(y >= -1e-14)
        assert np.all(s <= 1.0 + 1e-14)


def test_triangle_quadrature_2D_degree_exactness_1pt(fcn):
    """
    Accuracy of the 1-point centroid rule.

    This rule is exact for total degree ≤ 1. We verify exactness on monomials {1, x, y},
    then demonstrate non-exactness on representative quadratic monomials {x^2, xy, y^2}.
    Exact integral over the reference triangle T is: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    pts, w = fcn(1)
    x, y = pts[:, 0], pts[:, 1]

    # Exact up to degree 1
    for p, q in [(0, 0), (1, 0), (0, 1)]:
        exact = math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
        approx = np.dot(w, (x ** p) * (y ** q))
        assert np.isclose(approx, exact, rtol=1e-13, atol=1e-15)

    # Not exact for degree 2 (show at least one clear miss)
    for p, q in [(2, 0), (1, 1), (0, 2)]:
        exact = math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
        approx = np.dot(w, (x ** p) * (y ** q))
        # Should fail exactness with a noticeable error
        assert not np.isclose(approx, exact, rtol=1e-12, atol=1e-14)


def test_triangle_quadrature_2D_degree_exactness_3pt(fcn):
    """
    Accuracy of the classic 3-point rule.

    This rule is exact for total degree ≤ 2. We verify exactness on monomials
    {1, x, y, x^2, xy, y^2}, then demonstrate non-exactness on representative
    cubic monomials {x^3, x^2 y, x y^2, y^3}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    pts, w = fcn(3)
    x, y = pts[:, 0], pts[:, 1]

    # Exact up to degree 2
    for p, q in [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2)]:
        exact = math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
        approx = np.dot(w, (x ** p) * (y ** q))
        assert np.isclose(approx, exact, rtol=1e-12, atol=1e-14)

    # Not exact for degree 3
    for p, q in [(3,0), (2,1), (1,2), (0,3)]:
        exact = math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
        approx = np.dot(w, (x ** p) * (y ** q))
        assert not np.isclose(approx, exact, rtol=1e-12, atol=1e-14)


def test_triangle_quadrature_2D_degree_exactness_4pt(fcn):
    """
    Accuracy of the 4-point rule.

    This rule is exact for total degree ≤ 3. We verify exactness on all monomials with p+q ≤ 3,
    then demonstrate non-exactness on representative quartic monomials {x^4, x^3 y, x^2 y^2, x y^3, y^4}.
    Exact integral over T: ∫ x^p y^q dx dy = p! q! / (p+q+2)!.
    """
    pts, w = fcn(4)
    x, y = pts[:, 0], pts[:, 1]

    # Exact up to degree 3
    for p, q in [(0,0), (1,0), (0,1), (2,0), (1,1), (0,2), (3,0), (2,1), (1,2), (0,3)]:
        exact = math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
        approx = np.dot(w, (x ** p) * (y ** q))
        assert np.isclose(approx, exact, rtol=1e-11, atol=1e-13)

    # Not exact for degree 4
    for p, q in [(4,0), (3,1), (2,2), (1,3), (0,4)]:
        exact = math.factorial(p) * math.factorial(q) / math.factorial(p + q + 2)
        approx = np.dot(w, (x ** p) * (y ** q))
        assert not np.isclose(approx, exact, rtol=1e-11, atol=1e-13)


def task_info():
    task_id = "triangle_quadrature_2D"
    task_short_description = "returns quadrature points and weights for 2d triangular elements"
    created_date = "2025-09-22"
    created_by = "elejeune11"
    main_fcn = triangle_quadrature_2D
    required_imports = ["import math", "import numpy as np", "import pytest", "from typing import Tuple"]
    fcn_dependencies = []
    reference_verification_inputs = [[1], [3], [4]]
    test_cases = [{"test_code": test_triangle_quadrature_2D_invalid_inputs, "expected_failures": [triangle_quadrature_2D_expected_failure_no_error]},
                  {"test_code": test_triangle_quadrature_2D_basics, "expected_failures": [triangle_quadrature_2D_expected_failure_basics]},
                  {"test_code": test_triangle_quadrature_2D_degree_exactness_1pt, "expected_failures": [triangle_quadrature_2D_expected_failure_zeros, triangle_quadrature_2D_expected_failure_ones]},
                  {"test_code": test_triangle_quadrature_2D_degree_exactness_3pt, "expected_failures": [triangle_quadrature_2D_expected_failure_zeros, triangle_quadrature_2D_expected_failure_ones]},
                  {"test_code": test_triangle_quadrature_2D_degree_exactness_4pt, "expected_failures": [triangle_quadrature_2D_expected_failure_zeros, triangle_quadrature_2D_expected_failure_ones]}
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
