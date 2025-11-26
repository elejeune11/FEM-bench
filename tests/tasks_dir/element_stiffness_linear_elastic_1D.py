import numpy as np


def gauss_quadrature_1D(n: int) -> np.ndarray:
    """
    Return Gauss points and weights for 1D quadrature over [-1, 1].

    Parameters:
        n (int): Number of Gauss points (1, 2, or 3 recommended)

    Returns:
        np.ndarray[np.ndarray, np.ndarray]: (points, weights)
    """
    if n == 1:
        points = np.array([0.0])
        weights = np.array([2.0])
    elif n == 2:
        sqrt_3_inv = 1.0 / np.sqrt(3)
        points = np.array([-sqrt_3_inv, sqrt_3_inv])
        weights = np.array([1.0, 1.0])
    elif n == 3:
        points = np.array([
            -np.sqrt(3/5), 0.0, np.sqrt(3/5)
        ])
        weights = np.array([
            5/9, 8/9, 5/9
        ])
    else:
        raise ValueError("Only 1 to 3 Gauss points are supported.")
    
    return np.array([points, weights], dtype=object)


def shape_function_derivatives_1D_linear() -> np.ndarray:
    """
    Return constant derivatives of 1D linear shape functions w.r.t. ξ.

    Returns:
        np.ndarray: Derivatives [dN1/dξ, dN2/dξ]
    """
    return np.array([
        -0.5,
        0.5
    ])


def compute_jacobian_1D(dN_dxi: np.ndarray, x_elem: np.ndarray) -> float:
    """
    Compute the Jacobian for 1D isoparametric mapping.

    Parameters:
        dN_dxi (np.ndarray): Shape function derivatives w.r.t. ξ
        x_elem (np.ndarray): Nodal coordinates of the current element [x1, x2]

    Returns:
        float: Jacobian dx/dξ
    """
    return np.array(np.dot(dN_dxi, x_elem))


def element_stiffness_linear_elastic_1D(
    x_elem: np.ndarray,  # nodal coordinates [x1, x2]
    E: float,
    A: float,
    n_gauss: int = 2
) -> np.ndarray:
    """
    Compute the element stiffness matrix for a 1D linear bar using the Galerkin method.

    Parameters:
        x_elem (np.ndarray): Nodal coordinates of the element [x1, x2]
        E (float): Young's modulus
        A (float): Cross-sectional area
        n_gauss (int): Number of Gauss integration points (default = 2)

    Returns:
        np.ndarray: 2x2 element stiffness matrix
    """
    k_elem = np.zeros((2, 2))
    xi_points, weights = gauss_quadrature_1D(n_gauss)
    dN_dxi = shape_function_derivatives_1D_linear()

    for xi, w in zip(xi_points, weights):
        J = compute_jacobian_1D(dN_dxi, x_elem)
        dN_dx = dN_dxi / J  # physical shape function derivatives
        B = dN_dx.reshape(1, 2)  # row vector

        # k = ∫ Bᵀ * EA * B * J dξ ≈ Σ Bᵀ * EA * B * J * w
        k_elem += E * A * J * w * (B.T @ B)

    return k_elem


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
    # --- Setup: Define element and material properties ---
    x_elem = np.array([0.0, 2.0])  # Node positions; element length = 2.0
    E = 200e9                      # Young's modulus in Pascals
    A = 0.01                       # Cross-sectional area in m^2
    L = x_elem[1] - x_elem[0]      # Element length

    # --- 1. Analytical correctness ---
    K_computed = fcn(x_elem, E, A, n_gauss=2)
    K_expected = (E * A / L) * np.array([[1, -1], [-1, 1]])
    np.testing.assert_array_almost_equal(K_computed, K_expected, decimal=12)

    # --- 2. Shape and symmetry ---
    assert K_computed.shape == (2, 2), "Stiffness matrix must be 2x2"
    assert np.allclose(K_computed, K_computed.T), "Stiffness matrix must be symmetric"

    # --- 3. Singularity (due to rigid body motion of a single element) ---
    det_K = np.linalg.det(K_computed)
    assert np.isclose(det_K, 0.0, atol=1e-10), f"Expected singular matrix, got det = {det_K:.3e}"

    # --- 4. Integration consistency ---
    K1 = fcn(x_elem, E, A, n_gauss=1)
    K2 = fcn(x_elem, E, A, n_gauss=2)
    K3 = fcn(x_elem, E, A, n_gauss=3)

    np.testing.assert_array_almost_equal(K_computed, K1, decimal=12)
    np.testing.assert_array_almost_equal(K_computed, K2, decimal=12)
    np.testing.assert_array_almost_equal(K_computed, K3, decimal=12)


def wrong_element_stiffness_missing_area(
    x_elem: np.ndarray,
    E: float,
    A: float,
    num_gauss_pts: int = 2
) -> np.ndarray:
    """
    Incorrect stiffness matrix: missing area A in the integrand.
    """
    k_elem = np.zeros((2, 2))
    xi_points, weights = gauss_quadrature_1D(num_gauss_pts)
    dN_dxi = shape_function_derivatives_1D_linear()

    for xi, w in zip(xi_points, weights):
        J = compute_jacobian_1D(dN_dxi, x_elem)
        dN_dx = dN_dxi / J
        B = dN_dx.reshape(1, 2)

        # WRONG: missing area A
        k_elem += E * J * w * (B.T @ B)

    return k_elem


def wrong_element_stiffness_asymmetric(
    x_elem: np.ndarray,
    E: float,
    A: float,
    num_gauss_pts: int = 2
) -> np.ndarray:
    """
    Incorrect stiffness matrix: asymmetric due to sign error during assembly.
    Violates physical requirement of symmetry.
    """
    k_elem = np.zeros((2, 2))
    xi_points, weights = gauss_quadrature_1D(num_gauss_pts)
    dN_dxi = shape_function_derivatives_1D_linear()

    for xi, w in zip(xi_points, weights):
        J = compute_jacobian_1D(dN_dxi, x_elem)
        dN_dx = dN_dxi / J
        B = dN_dx.reshape(1, 2)

        k_integrand = E * A * J * w * (B.T @ B)

        # WRONG: introduces asymmetry
        k_elem[0, :] += k_integrand[0, :]
        k_elem[1, :] -= k_integrand[1, :]  # Wrong sign

    return k_elem


def task_info():
    task_id = "element_stiffness_linear_elastic_1D"
    task_short_description = "creates a 1D element stiffness matrix for linear elastic"
    created_date = "2025-07-11"
    created_by = "elejeune11"
    main_fcn = element_stiffness_linear_elastic_1D
    required_imports = ["import numpy as np", "import pytest"]
    fcn_dependencies = [gauss_quadrature_1D, shape_function_derivatives_1D_linear, compute_jacobian_1D]
    reference_verification_inputs = [
        [[0.0, 1.0], 200e9, 0.01, 2],
        [[-1.0, 1.0], 100e9, 0.02, 3],
        [[3.0, 6.0], 70e9, 0.005, 1],
        [[5.0, 2.0], 150e9, 0.008, 2],
        [[0.0, 10.0], 10e9, 0.05, 3]
    ]
    test_cases = [{"test_code": test_element_stiffness_comprehensive, "expected_failures": [wrong_element_stiffness_missing_area, wrong_element_stiffness_asymmetric]}]
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