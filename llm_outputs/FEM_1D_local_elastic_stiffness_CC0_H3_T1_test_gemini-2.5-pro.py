def test_element_stiffness_comprehensive(fcn):
    """Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.
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
    E = 210000000000.0
    A = 0.005
    x_elem = np.array([0.5, 2.0])
    L = x_elem[1] - x_elem[0]
    k_expected = E * A / L * np.array([[1.0, -1.0], [-1.0, 1.0]])
    k_computed_2pt = fcn(x_elem=x_elem, E=E, A=A, n_gauss=2)
    np.testing.assert_allclose(k_computed_2pt, k_expected, rtol=1e-12, atol=1e-12)
    assert k_computed_2pt.shape == (2, 2)
    np.testing.assert_allclose(k_computed_2pt, k_computed_2pt.T, rtol=1e-12, atol=1e-12)
    det_k = np.linalg.det(k_computed_2pt)
    assert np.isclose(det_k, 0.0, atol=1e-09)
    k_computed_1pt = fcn(x_elem=x_elem, E=E, A=A, n_gauss=1)
    k_computed_3pt = fcn(x_elem=x_elem, E=E, A=A, n_gauss=3)
    np.testing.assert_allclose(k_computed_1pt, k_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(k_computed_3pt, k_expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(k_computed_1pt, k_computed_2pt, rtol=1e-12, atol=1e-12)