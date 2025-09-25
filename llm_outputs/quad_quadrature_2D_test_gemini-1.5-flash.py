def test_quad_quadrature_2D_invalid_inputs(fcn):
    """Test that quad_quadrature_2D rejects invalid numbers of points.
    The quadrature rule only supports 1, 4, or 9 integration points.
    Any other request should raise a ValueError.
    """
    with pytest.raises(ValueError):
        fcn(num_pts=2)
    with pytest.raises(ValueError):
        fcn(num_pts=10)
    with pytest.raises(ValueError):
        fcn(num_pts=0)

def test_quad_quadrature_2D_basics(fcn):
    """Test basic structural properties of the quadrature rule for quads.
    For each supported rule (1, 4, 9 points):
      -1 <= x <= 1 and -1 <= y <= 1.
    """
    for num_pts in [1, 4, 9]:
        (points, weights) = fcn(num_pts)
        assert points.shape == (num_pts, 2)
        assert weights.shape == (num_pts,)
        assert points.dtype == np.float64
        assert weights.dtype == np.float64
        assert np.isclose(np.sum(weights), 4.0)
        assert np.all(np.abs(points) <= 1.0)

def test_quad_quadrature_2D_degree_exactness_1pt(fcn):
    """Validate the degree-exactness of the 1×1 Gauss–Legendre quadrature rule on the
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
    2. Adds quadratic contributions (x², y², xy).
         confirming that the 1-point rule is not exact for degree ≥ 2.
    Expected outcome
    ----------------
    • Exactness assertions for monomials of degree ≤ 1 should pass.
    • Non-exactness assertions for quadratics should fail the exactness check
      (i.e. the quadrature does not reproduce the analytic integrals).
    """
    (points, weights) = fcn(1)
    assert np.isclose(quadrature_result(fcn, 1, lambda x, y: 1), 4)
    assert np.isclose(quadrature_result(fcn, 1, lambda x, y: x), 0)
    assert np.isclose(quadrature_result(fcn, 1, lambda x, y: y), 0)
    assert not np.isclose(quadrature_result(fcn, 1, lambda x, y: x ** 2), 4 / 3)
    assert not np.isclose(quadrature_result(fcn, 1, lambda x, y: y ** 2), 4 / 3)
    assert not np.isclose(quadrature_result(fcn, 1, lambda x, y: x * y), 0)

def test_quad_quadrature_2D_degree_exactness_2x2(fcn):
    """Validate the degree-exactness of the 2×2 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
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
    assert np.isclose(quadrature_result(fcn, 4, lambda x, y: x ** 3), 0)
    assert np.isclose(quadrature_result(fcn, 4, lambda x, y: y ** 3), 0)
    assert np.isclose(quadrature_result(fcn, 4, lambda x, y: x ** 2 * y), 0)
    assert np.isclose(quadrature_result(fcn, 4, lambda x, y: x * y ** 2), 0)
    assert not np.isclose(quadrature_result(fcn, 4, lambda x, y: x ** 4), 16 / 5)
    assert not np.isclose(quadrature_result(fcn, 4, lambda x, y: y ** 4), 16 / 5)

def test_quad_quadrature_2D_degree_exactness_3x3(fcn):
    """Validate the degree-exactness of the 3×3 Gauss–Legendre quadrature rule on [-1,1]×[-1,1].
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
    assert np.isclose(quadrature_result(fcn, 9, lambda x, y: x ** 5), 0)
    assert np.isclose(quadrature_result(fcn, 9, lambda x, y: y ** 5), 0)
    assert np.isclose(quadrature_result(fcn, 9, lambda x, y: x ** 4 * y), 0)
    assert np.isclose(quadrature_result(fcn, 9, lambda x, y: x * y ** 4), 0)
    assert np.isclose(quadrature_result(fcn, 9, lambda x, y: x ** 3 * y ** 2), 0)
    assert np.isclose(quadrature_result(fcn, 9, lambda x, y: x ** 2 * y ** 3), 0)
    assert not np.isclose(quadrature_result(fcn, 9, lambda x, y: x ** 6), 64 / 7)
    assert not np.isclose(quadrature_result(fcn, 9, lambda x, y: y ** 6), 64 / 7)
    assert not np.isclose(quadrature_result(fcn, 9, lambda x, y: x ** 4 * y ** 2), 64 / 45)