def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
    This test checks the following properties of the stiffness matrix computed by
    `element_stiffness_linear_elastic_1D` for a two-node linear element:
    1. Analytical correctness:
          (EA/L) * [[1, -1], [-1, 1]]
    2. Shape and symmetry:
    3. Singularity:
          reflecting rigid body motion.
    4. Integration consistency:
          Gauss quadrature rules when applied to linear elements, since exact integration is achieved.
    Note: Minor floating-point differences may arise due to roundoff when summing weighted values.
    This test uses a strict but reasonable tolerance to allow for numerical consistency considering the limitations of floating point arithmetic.
    """
    x_elem = np.array([0.25, 3.75], dtype=float)
    E = 210.123456789
    A = 0.0123456789
    L = x_elem[1] - x_elem[0]
    expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]], dtype=float)
    K1 = fcn(x_elem, E, A, 1)
    K2 = fcn(x_elem, E, A, 2)
    K3 = fcn(x_elem, E, A, 3)
    assert K1.shape == (2, 2)
    assert np.allclose(K1, K1.T, rtol=1e-12, atol=1e-12)
    assert np.allclose(K1, expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(K2, expected, rtol=1e-12, atol=1e-12)
    assert np.allclose(K3, expected, rtol=1e-12, atol=1e-12)
    det = np.linalg.det(K1)
    scale = max(1.0, np.linalg.norm(K1, ord='fro') ** 2)
    assert abs(det) <= 1e-10 * scale
    assert np.allclose(K1, K2, rtol=1e-12, atol=1e-12)
    assert np.allclose(K1, K3, rtol=1e-12, atol=1e-12)
    assert np.allclose(K2, K3, rtol=1e-12, atol=1e-12)