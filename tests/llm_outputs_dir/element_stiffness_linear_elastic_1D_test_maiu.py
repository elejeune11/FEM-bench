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
    E = 210e9  # Young's modulus in Pascals
    A = 0.01   # Cross-sectional area in m^2
    x_elem = np.array([0.0, 2.0])  # Element from 0 to 2 meters
    L = x_elem[1] - x_elem[0]

    # Analytical stiffness matrix
    k_analytical = (E * A / L) * np.array([[1, -1], [-1, 1]])

    # Test for n_gauss = 1, 2, 3
    k_matrices = []
    for n_gauss in [1, 2, 3]:
        k = fcn(x_elem, E, A, n_gauss)
        k_matrices.append(k)

        # Check shape
        assert k.shape == (2, 2), f"Stiffness matrix shape is not 2x2 for n_gauss={n_gauss}"

        # Check symmetry
        assert np.allclose(k, k.T, atol=1e-12), f"Stiffness matrix not symmetric for n_gauss={n_gauss}"

        # Check analytical correctness within tolerance
        assert np.allclose(k, k_analytical, rtol=1e-8, atol=1e-12), f"Stiffness matrix incorrect for n_gauss={n_gauss}"

    # Check integration consistency: all matrices should be close to each other
    for i in range(len(k_matrices)-1):
        assert np.allclose(k_matrices[i], k_matrices[i+1], rtol=1e-12, atol=1e-14), \
            f"Stiffness matrices differ between n_gauss={i+1} and n_gauss={i+2}"

    # Check singularity: determinant should be zero or very close to zero
    det = np.linalg.det(k_matrices[0])
    assert np.isclose(det, 0.0, atol=1e-14), "Stiffness matrix determinant is not zero (should be singular)"