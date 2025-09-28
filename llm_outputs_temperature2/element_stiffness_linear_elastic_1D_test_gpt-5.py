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
    tol = 1e-12
    E = 123.456
    A = 7.89
    L = 2.345
    x1 = 1.1
    x_elem = np.array([x1, x1 + L], dtype=float)
    K_expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    K1 = fcn(x_elem, E, A, 1)
    K2 = fcn(x_elem, E, A, 2)
    K3 = fcn(x_elem, E, A, 3)
    assert K1.shape == (2, 2)
    assert K2.shape == (2, 2)
    assert K3.shape == (2, 2)
    assert np.allclose(K1, K1.T, rtol=0.0, atol=tol)
    assert np.allclose(K2, K2.T, rtol=0.0, atol=tol)
    assert np.allclose(K3, K3.T, rtol=0.0, atol=tol)
    assert np.allclose(K2, K_expected, rtol=0.0, atol=tol)
    assert np.allclose(K1, K2, rtol=0.0, atol=tol)
    assert np.allclose(K2, K3, rtol=0.0, atol=tol)
    det_K2 = np.linalg.det(K2)
    scale = np.linalg.norm(K2) ** 2
    assert abs(det_K2) <= tol * max(scale, 1.0)