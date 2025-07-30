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
    This test uses a strict but reasonable tolerance to allow for numerical consistency.
    """
    E = 200000000000.0
    A = 0.01
    x_elem = np.array([0, 1])
    L = x_elem[1] - x_elem[0]
    k_analytical = E * A / L * np.array([[1, -1], [-1, 1]])
    for n_gauss in [1, 2, 3]:
        k = fcn(x_elem, E, A, n_gauss)
        assert k.shape == (2, 2)
        assert np.allclose(k, k.T)
        assert np.isclose(np.linalg.det(k), 0)
        assert np.allclose(k, k_analytical)
        if n_gauss > 1:
            k_ref = fcn(x_elem, E, A, 2)
            assert np.allclose(k, k_ref, rtol=1e-12, atol=1e-12)