import numpy as np
import pytest

def test_element_stiffness_comprehensive(fcn):
    """
    Verify the correctness and robustness of the 1D linear elastic element stiffness matrix.

    This test checks the following properties of the stiffness matrix computed by
    `element_stiffness_linear_elastic_1D` for a two-node linear element:

    1. Analytical correctness:
        - For an element of length L with modulus E and area A, the stiffness matrix should match:
          (EA/L) * [[1, -1], [-1, 1]]

    2. Shape and symmetry:
        - The stiffness matrix must be 2x2 and symmetric.

    3. Singularity:
        - The matrix should be singular (zero determinant) for an unconstrained single element,
          reflecting rigid body motion.

    4. Integration consistency:
        - The result must be identical (within numerical tolerance) for 1-, 2-, and 3-point
          Gauss quadrature rules when applied to linear elements, since exact integration is achieved.
    """
    E = 210e9  # Young's modulus (Pa) - Steel
    A = 1e-4  # Cross-sectional area (m^2)
    x_elem = np.array([0.0, 1.0])  # Element coordinates (m)
    L = x_elem[1] - x_elem[0]

    # Analytical solution
    k_analytical = (E * A / L) * np.array([[1, -1], [-1, 1]])

    # Test for different Gauss point numbers
    for n_gauss in [1, 2, 3]:
        k = fcn(x_elem, E, A, n_gauss)

        # 1. Analytical correctness
        np.testing.assert_allclose(k, k_analytical)

        # 2. Shape and symmetry
        assert k.shape == (2, 2)
        assert np.allclose(k, k.T)

        # 3. Singularity
        assert np.isclose(np.linalg.det(k), 0.0)

        # 4. Integration consistency (check consistency between Gauss points, only for the final result after all Gauss points are checked)
        if n_gauss > 1:
           k_prev = fcn(x_elem, E, A, n_gauss -1)
           np.testing.assert_allclose(k, k_prev) # check against previous gauss point count.