import numpy as np


def local_elastic_stiffness_matrix_3D_beam(
    E: float,
    nu: float,
    A: float,
    L: float,
    Iy: float,
    Iz: float,
    J: float
) -> np.ndarray:
    """
    Return the 12×12 local elastic stiffness matrix for a 3D Euler-Bernoulli beam element.

    The beam is assumed to be aligned with the local x-axis. The stiffness matrix
    relates local nodal displacements and rotations to forces and moments using the equation:

        [force_vector] = [stiffness_matrix] @ [displacement_vector]

    Degrees of freedom are ordered as:
        [u1, v1, w1, θx1, θy1, θz1, u2, v2, w2, θx2, θy2, θz2]

    Where:
        - u, v, w: displacements along local x, y, z
        - θx, θy, θz: rotations about local x, y, z
        - Subscripts 1 and 2 refer to node i and node j of the element

    Parameters:
        E (float): Young's modulus
        nu (float): Poisson's ratio (used for torsion only)
        A (float): Cross-sectional area
        L (float): Length of the beam element
        Iy (float): Second moment of area about the local y-axis
        Iz (float): Second moment of area about the local z-axis
        J (float): Torsional constant

    Returns:
        np.ndarray: A 12×12 symmetric stiffness matrix representing axial, torsional,
                    and bending stiffness in local coordinates.
    """
    k_e = np.zeros((12, 12))
    # Axial terms - extension of local x axis
    axial_stiffness = E * A / L
    k_e[0, 0] = axial_stiffness
    k_e[0, 6] = -axial_stiffness
    k_e[6, 0] = -axial_stiffness
    k_e[6, 6] = axial_stiffness
    # Torsion terms - rotation about local x axis
    torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
    k_e[3, 3] = torsional_stiffness
    k_e[3, 9] = -torsional_stiffness
    k_e[9, 3] = -torsional_stiffness
    k_e[9, 9] = torsional_stiffness
    # Bending terms - bending about local z axis
    k_e[1, 1] = E * 12.0 * Iz / L ** 3.0
    k_e[1, 7] = E * -12.0 * Iz / L ** 3.0
    k_e[7, 1] = E * -12.0 * Iz / L ** 3.0
    k_e[7, 7] = E * 12.0 * Iz / L ** 3.0
    k_e[1, 5] = E * 6.0 * Iz / L ** 2.0
    k_e[5, 1] = E * 6.0 * Iz / L ** 2.0
    k_e[1, 11] = E * 6.0 * Iz / L ** 2.0
    k_e[11, 1] = E * 6.0 * Iz / L ** 2.0
    k_e[5, 7] = E * -6.0 * Iz / L ** 2.0
    k_e[7, 5] = E * -6.0 * Iz / L ** 2.0
    k_e[7, 11] = E * -6.0 * Iz / L ** 2.0
    k_e[11, 7] = E * -6.0 * Iz / L ** 2.0
    k_e[5, 5] = E * 4.0 * Iz / L
    k_e[11, 11] = E * 4.0 * Iz / L
    k_e[5, 11] = E * 2.0 * Iz / L
    k_e[11, 5] = E * 2.0 * Iz / L
    # Bending terms - bending about local y axis
    k_e[2, 2] = E * 12.0 * Iy / L ** 3.0
    k_e[2, 8] = E * -12.0 * Iy / L ** 3.0
    k_e[8, 2] = E * -12.0 * Iy / L ** 3.0
    k_e[8, 8] = E * 12.0 * Iy / L ** 3.0
    k_e[2, 4] = E * -6.0 * Iy / L ** 2.0
    k_e[4, 2] = E * -6.0 * Iy / L ** 2.0
    k_e[2, 10] = E * -6.0 * Iy / L ** 2.0
    k_e[10, 2] = E * -6.0 * Iy / L ** 2.0
    k_e[4, 8] = E * 6.0 * Iy / L ** 2.0
    k_e[8, 4] = E * 6.0 * Iy / L ** 2.0
    k_e[8, 10] = E * 6.0 * Iy / L ** 2.0
    k_e[10, 8] = E * 6.0 * Iy / L ** 2.0
    k_e[4, 4] = E * 4.0 * Iy / L
    k_e[10, 10] = E * 4.0 * Iy / L
    k_e[4, 10] = E * 2.0 * Iy / L
    k_e[10, 4] = E * 2.0 * Iy / L
    return k_e


def test_local_stiffness_3D_beam(fcn):
    """
    Comprehensive test for local_elastic_stiffness_matrix_3D_beam:
    - shape check
    - symmetry
    - expected singularity due to rigid body modes
    - block-level verification of axial, torsion, and bending terms
    """
    # Beam properties
    E = 200e9         # Young's modulus
    nu = 0.3          # Poisson's ratio
    A = 0.01          # Cross-sectional area
    L = 2.0           # Length of the beam
    Iy = 8e-6         # Moment of inertia about y
    Iz = 6e-6         # Moment of inertia about z
    J = 1e-5          # Torsional constant

    k = fcn(E, nu, A, L, Iy, Iz, J)

    # --- Shape check ---
    assert k.shape == (12, 12)

    # --- Symmetry check ---
    assert np.allclose(k, k.T, atol=1e-12)

    # --- Singularity check (due to 6 rigid-body modes) ---
    eigvals = np.linalg.eigvalsh(k)
    num_near_zero = np.sum(np.isclose(eigvals, 0.0, atol=1e-8))
    assert num_near_zero >= 6, f"Expected at least 6 near-zero eigenvalues, got {num_near_zero}"

    # --- Axial terms block ---
    expected_axial = E * A / L
    assert np.isclose(k[0, 0], expected_axial, rtol=1e-12)
    assert np.isclose(k[0, 6], -expected_axial, rtol=1e-12)
    assert np.isclose(k[6, 0], -expected_axial, rtol=1e-12)
    assert np.isclose(k[6, 6], expected_axial, rtol=1e-12)

    # --- Torsional terms block (theta_x DOFs) ---
    G = E / (2 * (1 + nu))
    expected_torsion = G * J / L
    assert np.isclose(k[3, 3], expected_torsion, rtol=1e-12)
    assert np.isclose(k[3, 9], -expected_torsion, rtol=1e-12)
    assert np.isclose(k[9, 3], -expected_torsion, rtol=1e-12)
    assert np.isclose(k[9, 9], expected_torsion, rtol=1e-12)

    # --- Bending about local z (v–theta_z: DOFs 1, 5, 7, 11) ---
    expected_bz_11 = E * 12.0 * Iz / L**3
    expected_bz_15 = E * 6.0 * Iz / L**2
    expected_bz_55 = E * 4.0 * Iz / L
    expected_bz_511 = E * 2.0 * Iz / L
    assert np.isclose(k[1, 1], expected_bz_11, rtol=1e-12)
    assert np.isclose(k[1, 5], expected_bz_15, rtol=1e-12)
    assert np.isclose(k[5, 5], expected_bz_55, rtol=1e-12)
    assert np.isclose(k[5, 11], expected_bz_511, rtol=1e-12)

    # --- Bending about local y (w–theta_y: DOFs 2, 4, 8, 10) ---
    expected_by_22 = E * 12.0 * Iy / L**3
    expected_by_24 = -E * 6.0 * Iy / L**2
    expected_by_44 = E * 4.0 * Iy / L
    expected_by_410 = E * 2.0 * Iy / L
    assert np.isclose(k[2, 2], expected_by_22, rtol=1e-12)
    assert np.isclose(k[2, 4], expected_by_24, rtol=1e-12)
    assert np.isclose(k[4, 4], expected_by_44, rtol=1e-12)
    assert np.isclose(k[4, 10], expected_by_410, rtol=1e-12)


def local_elastic_stiffness_matrix_3D_beam_flipped_Iz_Iy(
    E: float,
    nu: float,
    A: float,
    L: float,
    Iy: float,
    Iz: float,
    J: float
) -> np.ndarray:
    k_e = np.zeros((12, 12))
    # Axial terms - extension of local x axis
    axial_stiffness = E * A / L
    k_e[0, 0] = axial_stiffness
    k_e[0, 6] = -axial_stiffness
    k_e[6, 0] = -axial_stiffness
    k_e[6, 6] = axial_stiffness
    # Torsion terms - rotation about local x axis
    torsional_stiffness = E * J / (2.0 * (1 + nu) * L)
    k_e[3, 3] = torsional_stiffness
    k_e[3, 9] = -torsional_stiffness
    k_e[9, 3] = -torsional_stiffness
    k_e[9, 9] = torsional_stiffness
    # Bending terms - bending about local z axis
    k_e[1, 1] = E * 12.0 * Iy / L ** 3.0
    k_e[1, 7] = E * -12.0 * Iy / L ** 3.0
    k_e[7, 1] = E * -12.0 * Iy / L ** 3.0
    k_e[7, 7] = E * 12.0 * Iy / L ** 3.0
    k_e[1, 5] = E * 6.0 * Iy / L ** 2.0
    k_e[5, 1] = E * 6.0 * Iy / L ** 2.0
    k_e[1, 11] = E * 6.0 * Iy / L ** 2.0
    k_e[11, 1] = E * 6.0 * Iy / L ** 2.0
    k_e[5, 7] = E * -6.0 * Iy / L ** 2.0
    k_e[7, 5] = E * -6.0 * Iy / L ** 2.0
    k_e[7, 11] = E * -6.0 * Iy / L ** 2.0
    k_e[11, 7] = E * -6.0 * Iy / L ** 2.0
    k_e[5, 5] = E * 4.0 * Iy / L
    k_e[11, 11] = E * 4.0 * Iy / L
    k_e[5, 11] = E * 2.0 * Iy / L
    k_e[11, 5] = E * 2.0 * Iy / L
    # Bending terms - bending about local y axis
    k_e[2, 2] = E * 12.0 * Iz / L ** 3.0
    k_e[2, 8] = E * -12.0 * Iz / L ** 3.0
    k_e[8, 2] = E * -12.0 * Iz / L ** 3.0
    k_e[8, 8] = E * 12.0 * Iz / L ** 3.0
    k_e[2, 4] = E * -6.0 * Iz / L ** 2.0
    k_e[4, 2] = E * -6.0 * Iz / L ** 2.0
    k_e[2, 10] = E * -6.0 * Iz / L ** 2.0
    k_e[10, 2] = E * -6.0 * Iz / L ** 2.0
    k_e[4, 8] = E * 6.0 * Iz / L ** 2.0
    k_e[8, 4] = E * 6.0 * Iz / L ** 2.0
    k_e[8, 10] = E * 6.0 * Iz / L ** 2.0
    k_e[10, 8] = E * 6.0 * Iz / L ** 2.0
    k_e[4, 4] = E * 4.0 * Iz / L
    k_e[10, 10] = E * 4.0 * Iz / L
    k_e[4, 10] = E * 2.0 * Iz / L
    k_e[10, 4] = E * 2.0 * Iz / L
    return k_e


def task_info():
    task_id = "local_elastic_stiffness_matrix_3D_beam"
    task_short_description = "creates an element stiffness matrix for a 3D beam"
    created_date = "2025-07-31"
    created_by = "elejeune11"
    main_fcn = local_elastic_stiffness_matrix_3D_beam
    required_imports = ["import numpy as np", "import pytest"]
    fcn_dependencies = []
    reference_verification_inputs = [[100, 0.3, 10, 5, 30, 25, 10],
                                     [10000, 0.4, 77, 55, 300, 250, 9.9],
                                     [98000, 0.3, 5.5, 55, 300, 250, 9.4],
                                     [6790, 0.2, 10.6, 4.7, 44, 34, 20.1],]
    test_cases = [{"test_code": test_local_stiffness_3D_beam, "expected_failures": [local_elastic_stiffness_matrix_3D_beam_flipped_Iz_Iy]}]
    return task_id, task_short_description, created_date, created_by, main_fcn, required_imports, fcn_dependencies, reference_verification_inputs, test_cases
